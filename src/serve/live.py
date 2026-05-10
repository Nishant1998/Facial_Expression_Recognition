"""One pending frame per connection; newer frames replace stale work."""
import asyncio

import cv2
import numpy as np


class LatestFrame:
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=1)

    def put(self, frame):
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(frame)

    async def get(self):
        return await self.queue.get()


def detect_frame(data, model, threshold):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Invalid camera frame.')
    h, w = frame.shape[:2]
    if h * w > 1280 * 1280:
        raise ValueError('Camera frame is too large.')
    result = model(frame, threshold)
    return [dict(
        bbox=[float(d['bbox'][0]) / w, float(d['bbox'][1]) / h,
              float(d['bbox'][2]) / w, float(d['bbox'][3]) / h],
        label=d['emotion_label'], emoji=d['emoji'],
    ) for d in result['detections']]
