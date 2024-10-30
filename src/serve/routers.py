import asyncio
import json
import os
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
from starlette.requests import ClientDisconnect

from src.serve.inferencer import process_image_with_model
from src.utils import setup_logger

logger = setup_logger()
router = APIRouter()


# === Utility Functions ===
def log_info(client: str, message: str):
    logger.info(f"[USER: {client}] {message}")

def validate_file(file: UploadFile, allowed_exts: set, max_mb: int):
    ext = os.path.splitext(file.filename)[-1].lower()
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
    cfg = request.app.state.cfg
    client_ip = request.client.host

    try:
        log_info(client_ip, f"Image upload: {file.filename} | Emoji: {emoji}")
        validate_file(file, set(cfg.API.ALLOWED_IMAGE_EXT), cfg.API.MAX_IMAGE_SIZE_MB)

        contents = await file.read()
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
        processed = process_image_with_model(image, model, cfg)

        _, jpeg = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

        log_info(client_ip, "Image processed")
        return StreamingResponse(BytesIO(jpeg.tobytes()), media_type="image/jpeg")

    except Exception as e:
        log_info(client_ip, f"Image processing failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    cfg = websocket.app.state.cfg
    await websocket.accept()
    client_ip = websocket.client.host
    frame_id = 0

    log_info(client_ip, "WebSocket connected")

    try:
        while True:
            message = await websocket.receive()

            # === Handle config messages (text JSON)
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        cfg.DISPLAY.EMOJI = data.get("emoji", False)
                        log_info(client_ip, f"Emoji mode updated: {cfg.DISPLAY.EMOJI}")
                except json.JSONDecodeError:
                    log_info(client_ip, "Invalid JSON config received")
                continue

            # === Handle image frame (binary JPEG)
            elif "bytes" in message:
                data = message["bytes"]
                npimg = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

                if frame is None:
                    log_info(client_ip, f"Invalid frame {frame_id}")
                    continue

                model = websocket.app.state.model
                processed = process_image_with_model(frame, model, cfg)
                _, jpeg = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                await websocket.send_bytes(jpeg.tobytes())

                if frame_id % 10 == 0:
                    log_info(client_ip, f"Sent frame {frame_id}")
                frame_id += 1

                await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        log_info(client_ip, "WebSocket disconnected")
    except ClientDisconnect:
        log_info(client_ip, "Client forcefully closed")
    except Exception as e:
        log_info(client_ip, f"WebSocket error: {str(e)}")
