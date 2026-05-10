"""Match the RGB, resize and normalization used by held-out evaluation."""
import cv2
import numpy as np
from PIL import Image


def prepare_face(bgr, size=128):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    pixels = (pixels - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return pixels.transpose(2, 0, 1).copy()
