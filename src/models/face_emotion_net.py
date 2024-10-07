import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src.models.efficientnet_fer import EfficientNetFER
from src.utils import setup_logger

logger = setup_logger()


class FaceEmotionNet:
    def __init__(self, yolo_model='models/yolov11n-face.onnx', fer_model='models/EfficientNetFER.onnx', verbose=True):
        """
        Initializes the FaceEmotionNet model.

        :param yolo_model: Path to YOLO model for face detection.
        :param fer_model: Path to PyTorch or ONNX EfficientNetFER model.
        :param verbose: If True, enables detailed logging; if False, only essential logs are printed.
        """
        self.verbose = verbose

        if self.verbose:
            logger.info("Initializing FaceEmotionNet model...")

        try:
            self.Yolo = YOLO(yolo_model)
            if self.verbose:
                logger.info(f"YOLO model loaded successfully from {yolo_model}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}", exc_info=True)
            raise

        available_providers = ort.get_available_providers()
        use_gpu = "CUDAExecutionProvider" in available_providers
        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

        try:
            self.FerNet = EfficientNetFER(fer_model, verbose=self.verbose)
            if self.verbose:
                logger.info(f"ONNX model loaded successfully from {fer_model} using {'GPU' if use_gpu else 'CPU'}")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}", exc_info=True)
            raise

        # Emotion labels & emojis
        self.classes = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral'}
        self.emotion_emojis = {0: "1f632", 1: "1f628", 2: "1f922", 3: "1f60a", 4: "1f622", 5: "1f621", 6: "1f610"}

        if self.verbose:
            logger.info("FaceEmotionNet initialized successfully.")

    def __call__(self, image, confidence_threshold=0.5):
        """
        Runs face detection and emotion recognition.

        :param image: Input image (NumPy array).
        :param confidence_threshold: Minimum confidence to consider detections.
        :return: Dictionary with bounding boxes, labels, and emojis.
        """
        if self.verbose:
            logger.debug("Running face detection on input image...")

        result = self.Yolo(image, verbose=False)[0]

        # Extract face crops and bounding boxes
        face_images, bbox = self.extract_faces_from_yolo(result, image)

        if len(face_images) == 0:
            if self.verbose:
                logger.info("No faces detected.")
            return {"detections": [], "image": result.orig_img}

        if self.verbose:
            logger.debug(f"Running emotion recognition on {len(face_images)} detected faces.")

        try:
            class_ids = self.FerNet(face_images)
        except Exception as e:
            logger.error(f"Error during emotion recognition: {e}", exc_info=True)
            return {"detections": [], "image": result.orig_img}

        detections = []
        for i, (x1, y1, x2, y2, conf, class_label) in enumerate(bbox):
            if float(conf) < confidence_threshold or class_label != "face":
                continue  # Skip detections below confidence threshold

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(float(conf), 2),
                "class_id": class_ids[i],
                "class": "face",
                "emotion_label": self.classes[class_ids[i]],
                "emoji": self.emotion_emojis[class_ids[i]]
            })

        if self.verbose:
            logger.info(f"Detected {len(detections)} faces with emotions.")

        return {"detections": detections, "image": result.orig_img}

    def plot(self, result, cfg, fps=None):
        """
        Visualizes the detected faces and their emotions with correct emoji display.

        :param result: Dictionary containing detections and image.
        :param cfg: Configuration settings.
        :param fps: Frame rate for overlay display.
        :return: Annotated image.
        """
        display_fps = cfg.DISPLAY.FPS
        color = cfg.DISPLAY.OVERLAY_COLOR

        detections = result['detections']
        image = result['image']

        if self.verbose:
            logger.debug(f"Plotting {len(detections)} detections on the image.")

        # Convert OpenCV image (BGR) to PIL image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.load_default(32)

            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                emotion_label = detection["emotion_label"]
                emoji = detection["emoji"]
                confidence = detection["confidence"]

                text = f"{emotion_label}"
                if cfg.DISPLAY.EMOJI:
                    try:
                        emoji_path = os.path.join("src/asset/emoji/", emoji.lower() + ".png")
                        if os.path.exists(emoji_path):
                            emoji_img = Image.open(emoji_path).convert("RGBA")
                            w, h = int(x2 - x1), int(y2 - y1)
                            emoji_resized = emoji_img.resize((w, h), Image.Resampling.LANCZOS)
                            pil_image.paste(emoji_resized, (int(x1), int(y1)), emoji_resized)
                        else:
                            draw.text((x1, y1 - 30), text, font=font, fill=color)
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    except Exception as e:
                        logger.warning(f"Could not draw emoji {emoji}: {e}")
                        draw.text((x1, y1 - 30), text, font=font, fill=color)
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                else:
                    draw.text((x1, y1 - 30), text, font=font, fill=color)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            if display_fps and fps is not None:
                draw.text((10, 30), f"FPS: {int(fps)}", font=font, fill=color)

            if self.verbose:
                logger.debug("Annotations completed successfully.")

        except Exception as e:
            logger.error(f"Error rendering emoji: {e}", exc_info=True)

        # Convert PIL image back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def extract_faces_from_yolo(self, result, image, target_size=128):
        """
        Extracts face bounding boxes and crops faces into a square format.

        :param result: YOLO result object.
        :param image: The original image (NumPy array) from which faces are cropped.
        :param target_size: The target size for face recognition models (default: 128x128).
        :return: NumPy array of resized face images and their bounding boxes.
        """
        if result.boxes is None or len(result.boxes) == 0:
            if self.verbose:
                logger.info("No faces detected in the image.")
            return np.array([]), np.array([])

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        class_indices = result.boxes.cls.cpu().numpy()

        face_images, face_bboxes = [], []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            confidence = round(float(confidences[i]), 2)
            class_label = result.names[int(class_indices[i])]

            if class_label != "face":
                continue

            face_crop = image[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            face_images.append(face_resized)
            face_bboxes.append([x1, y1, x2, y2, confidence, class_label])

        face_images = np.array(face_images, dtype=np.float32) / 255.0
        face_images = np.transpose(face_images, (0, 3, 1, 2))

        if self.verbose:
            logger.info(f"Extracted {len(face_images)} face(s) for emotion detection.")

        return face_images, np.array(face_bboxes)
