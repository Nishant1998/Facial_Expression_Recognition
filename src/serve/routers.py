import asyncio
import json
import os

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
