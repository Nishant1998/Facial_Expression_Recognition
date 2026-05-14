"""Modal deployment entry point for the Facial Expression Recognition demo.

Deploy with:
    modal deploy modal_app.py

This file is intentionally separate from the Render configuration. It exposes
the existing FastAPI application on a Modal CPU-backed ASGI endpoint.
"""

import modal


app = modal.App("facial-expression-recognition")

# The project needs OpenCV system libraries and a CPU-only inference runtime.
# Model weights are downloaded while this image is built, so a visitor's
# request only needs to start the already-prepared container.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 "
        "--index-url https://download.pytorch.org/whl/cpu"
    )
    .pip_install(
        "timm==1.0.15",
        "onnxruntime==1.21.0",
        "onnx==1.17.0",
        "opencv-python-headless==4.11.0.86",
        "numpy==2.1.1",
        "fastapi==0.115.12",
        "uvicorn==0.34.3",
        "yacs==0.1.8",
        "emoji==2.14.1",
        "tqdm==4.67.1",
        "Pillow==11.1.0",
        "ultralytics==8.3.94",
        "starlette==0.46.2",
        "python-multipart",
        "websockets",
    )
    .env(
        {
            "PYTHONPATH": "/root",
            "APP_CONFIG": "src/config/custom.yaml",
            "FER_DEVICE": "cpu",
            "YOLO_AUTOINSTALL": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("frontend", remote_path="/root/frontend", copy=True)
    .run_commands(
        "python -c \"from src.utils.config_utils import load_config; from src.utils import download_weights; import onnx; cfg = load_config('src/config/custom.yaml'); [(download_weights(path, url), onnx.checker.check_model(path)) for path, url in [(cfg.INFERENCE.YOLO_PATH, cfg.INFERENCE.YOLO_DOWNLOAD_URL), (cfg.INFERENCE.FER_PATH, cfg.INFERENCE.FER_DOWNLOAD_URL)]]\""
    )
)


@app.function(
    image=image,
    cpu=1.0,
    memory=(2048, 2048),
    max_containers=1,
    scaledown_window=60,
    timeout=300,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fer_web_app():
    """Return the existing FastAPI app inside the CPU inference container."""
    import numpy as np

    from src.serve.app import app as fastapi_app

    # Initialize Ultralytics' lazy ONNX session before the first visitor's
    # request. The blank image has no face, so no artificial result is served.
    fastapi_app.state.model(np.zeros((128, 128, 3), dtype=np.uint8))
    return fastapi_app
