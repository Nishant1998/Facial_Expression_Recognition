import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from src.models import FaceEmotionNet
from src.utils import setup_logger

logger = setup_logger()


def infer(cfg):
    """
    Main function to run facial emotion recognition.

    :param cfg: Configuration object loaded from YAML or default config.
    """
    mode = cfg.VIDEO.MODE
    video_path = cfg.VIDEO.VIDEO_PATH
    webcam_id = cfg.VIDEO.WEBCAM_ID
    overlay_color = tuple(cfg.DISPLAY.OVERLAY_COLOR)

    logger.info("Initializing FaceEmotionNet...")
    logger.info(f"Running in {mode} mode")

    # Load the model
    model = FaceEmotionNet(cfg.INFERENCE.YOLO_PATH, cfg.INFERENCE.FER_PATH, cfg.DISPLAY.VERBOSE)
    logger.info("Model loaded successfully.")

    # Open video source (webcam or video file)
    cap = cv2.VideoCapture(webcam_id if mode == "webcam" else video_path)

    if not cap.isOpened():
        logger.error(
            f"Failed to open {mode}. Check webcam ID or video path: {video_path if mode == 'video' else webcam_id}")
        return

    logger.info("Video stream opened successfully.")

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame. Exiting loop.")
            break

        logger.debug("Frame captured successfully. Running inference...")

        # Perform inference
        curr_time = time.time()

        try:
            result = model(frame, cfg.INFERENCE.CONFIDENCE_THRESHOLD)
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            annotated_frame = model.plot(result, cfg, fps)

            cv2.imshow('YOLOv11 Face Detection', annotated_frame)

        except Exception as e:
            logger.error(f"Error during inference or plotting: {e}", exc_info=True)
            break

        # Check for user exit key press
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:  # 'q' or 'Esc'
            logger.info("Exit key pressed. Closing application.")
            break

    # Release video source and close all windows
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Video stream closed and resources released.")


def process_image_with_model(image: np.ndarray, model, cfg) -> np.ndarray:
    """
    Run prediction on a single image frame and return annotated output.
    """
    try:
        result = model(image, cfg.INFERENCE.CONFIDENCE_THRESHOLD)
        annotated = model.plot(result, cfg, fps=0)
        return annotated
    except Exception as e:
        logger.error(f"[IMAGE] Inference failed: {e}")
        raise



