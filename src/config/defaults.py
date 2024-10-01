from yacs.config import CfgNode as CN

_C = CN()

# Video processing settings
_C.VIDEO = CN()
_C.VIDEO.MODE = "webcam"  # webcam, video
_C.VIDEO.VIDEO_PATH = "src/data/data/datasample_video.mp4"
_C.VIDEO.WEBCAM_ID = 0



# ===========================
# 🔹 MODEL SETTINGS
# ===========================
_C.MODEL = CN()


# ===========================
# 🔹 LOGGING SETTINGS
# ===========================
_C.LOGGING = CN()
_C.LOGGING.SAVE_TO_FILE = True  # Whether to save logs to a file
_C.LOGGING.LOG_FILE = "logs/output.log"  # Log file path

# ===========================
# 🔹 DATASET SETTINGS
# ===========================
_C.DATASET = CN()
_C.DATASET.PATH = "src/data/data"

# ===========================
# 🔹 TRAINING SETTINGS
# ===========================
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LR = 0.001
_C.TRAIN.EPOCHS = 1
_C.TRAIN.MODEL_NAME = 'efficientnet_b0'

# ===========================
# 🔹 EXPORT SETTINGS
# ===========================
_C.EXPORT = CN()
_C.EXPORT.ENABLED = True
_C.EXPORT.EVAL = True
_C.EXPORT.MODEL_NAME = 'efficientnet_b0'
_C.EXPORT.INPUT_WEIGHTS = "src/models/weights/efficientnet_b0_fer.pth"  # original .pt weights
_C.EXPORT.OUTPUT_PATH = "src/models/weights/EfficientNetFER.onnx"                   # ONNX or TRT path

# ===========================
# 🔹 INFERENCE SETTINGS
# ===========================
_C.INFERENCE = CN()
_C.INFERENCE.CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
_C.INFERENCE.YOLO_PATH = "src/models/weights/yolov11n-face.onnx"  # YOLO face detection model path
_C.INFERENCE.FER_PATH = "src/models/weights/EfficientNetFER.onnx"  # Facial Expression Recognition (FER) model path



_C.IS_TRAINING = True




