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


def process_video_with_model(input_path: str, output_path: str, model, cfg) -> None:
    """
    Read video file, apply model inference per frame, save annotated video.
    """
    logger.info(f"[VIDEO] Opening video: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        logger.error(f"[VIDEO] Failed to open video: {input_path}")
        raise IOError(f"Cannot open video file: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tempfile.TemporaryDirectory() as temp_frames_dir:
        logger.info(f"[VIDEO] Temp frame storage: {temp_frames_dir}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                result = model(frame, cfg.INFERENCE.CONFIDENCE_THRESHOLD)
                processed = model.plot(result, cfg, fps=fps)
            except Exception as e:
                logger.warning(f"[VIDEO] Frame {frame_idx} processing failed: {e}")
                processed = frame  # fallback: keep original frame

            frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, processed, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

            if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                logger.info(f"[VIDEO] Frame {frame_idx + 1}/{total_frames} processed")

            frame_idx += 1

        cap.release()

        # Build ffmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_frames_dir, "frame_%04d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        logger.info("[VIDEO] Encoding video using ffmpeg...")

        try:
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"[VIDEO] Successfully wrote processed video to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"[VIDEO] FFmpeg failed: {e}")
            raise RuntimeError("Video encoding failed via ffmpeg.")

