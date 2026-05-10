import asyncio
import os
import tempfile
import time
import contextlib
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
    Request,
    HTTPException,
    Form,
)
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from starlette.background import BackgroundTask

from src.serve.inferencer import process_video_with_model, process_image_with_model
from src.utils import setup_logger
from src.serve.live import LatestFrame, detect_frame

logger = setup_logger()
router = APIRouter()


# === Utility Functions ===
def log_info(client: str, message: str):
    logger.info(f"[USER: {client}] {message}")

def validate_file(file: UploadFile, allowed_exts: set, max_mb: int):
    ext = os.path.splitext(file.filename or '')[-1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed.")
    if hasattr(file, "size") and file.size:
        if (file.size / (1024 * 1024)) > max_mb:
            raise HTTPException(status_code=400, detail=f"File too large (>{max_mb} MB).")

# === Routes ===
@router.post("/predict/image")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    emoji: Optional[bool] = Form(False)
):
    cfg = request.app.state.cfg.clone()
    client_ip = request.client.host

    try:
        log_info(client_ip, f"Image upload: {file.filename} | Emoji: {emoji}")
        validate_file(file, set(cfg.API.ALLOWED_IMAGE_EXT), cfg.API.MAX_IMAGE_SIZE_MB)

        contents = await file.read(cfg.API.MAX_IMAGE_SIZE_MB * 1024 * 1024 + 1)
        if len(contents) > cfg.API.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail='Image too large.')
        if not contents:
            raise HTTPException(status_code=400, detail='Empty image.')
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

        if image is None:
            log_info(client_ip, "Invalid image data")
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})
        else:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        cfg.DISPLAY.EMOJI = emoji

        model = request.app.state.model
        processed = await run_in_threadpool(process_image_with_model, image, model, cfg)

        _, jpeg = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

        log_info(client_ip, "Image processed")
        return StreamingResponse(BytesIO(jpeg.tobytes()), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        log_info(client_ip, f"Image processing failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/predict/video")
async def predict_video(request: Request, file: UploadFile = File(...), emoji: Optional[bool] = Form(False)):
    cfg = request.app.state.cfg.clone()
    client_ip = request.client.host
    workdir = None
    try:
        log_info(client_ip, f"Video upload: {file.filename} | Emoji: {emoji}")
        validate_file(file, set(cfg.API.ALLOWED_VIDEO_EXT), cfg.API.MAX_VIDEO_SIZE_MB)

        video_bytes = await file.read(cfg.API.MAX_VIDEO_SIZE_MB * 1024 * 1024 + 1)
        size_mb = len(video_bytes) / (1024 * 1024)
        if size_mb > cfg.API.MAX_VIDEO_SIZE_MB:
            raise HTTPException(status_code=400, detail="Video too large.")

        workdir = tempfile.TemporaryDirectory(prefix='fer-video-')
        input_path = os.path.join(workdir.name, 'input.mp4')
        output_path = os.path.join(workdir.name, 'output.mp4')
        with open(input_path, 'wb') as temp_input:
            temp_input.write(video_bytes)

        log_info(client_ip, f"Processing video ({size_mb:.2f} MB)...")
        cfg.DISPLAY.EMOJI = emoji
        start = time.time()
        model = request.app.state.model
        await run_in_threadpool(process_video_with_model, input_path, output_path, model, cfg)
        log_info(client_ip, f"Video done in {time.time() - start:.2f} sec")

        return FileResponse(output_path, media_type="video/mp4", filename="processed_video.mp4", background=BackgroundTask(workdir.cleanup))

    except HTTPException:
        if workdir:
            workdir.cleanup()
        raise
    except (ValueError, OSError) as e:
        if workdir:
            workdir.cleanup()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        if workdir:
            workdir.cleanup()
        log_info(client_ip, f"Video processing failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    cfg = websocket.app.state.cfg.clone()
    await websocket.accept()
    frames = LatestFrame()
    send_lock = asyncio.Lock()

    async def send(payload):
        async with send_lock:
            await websocket.send_json(payload)

    async def process_latest():
        while True:
            frame_id, data = await frames.get()
            started = time.perf_counter()
            try:
                detections = await run_in_threadpool(detect_frame, data, websocket.app.state.model, cfg.INFERENCE.CONFIDENCE_THRESHOLD)
                await send({'type': 'result', 'frame_id': frame_id, 'detections': detections,
                            'processing_ms': round((time.perf_counter() - started) * 1000)})
            except Exception:
                logger.exception('Camera inference failed')
                await send({'type': 'error', 'frame_id': frame_id, 'message': 'Camera frame could not be processed.'})

    worker = asyncio.create_task(process_latest())
    try:
        while True:
            # Four-byte big-endian frame ID followed by a JPEG. ACK means received,
            # not processed, allowing capture/network work to overlap inference.
            packet = await websocket.receive_bytes()
            if len(packet) <= 4 or len(packet) > 2 * 1024 * 1024:
                await websocket.close(code=1009)
                break
            frame_id = int.from_bytes(packet[:4], 'big')
            frames.put((frame_id, packet[4:]))
            await send({'type': 'received', 'frame_id': frame_id})
    except (WebSocketDisconnect, ClientDisconnect):
        pass
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
