import argparse
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.models import FaceEmotionNet
from src.serve.routers import router
from src.utils import setup_logger
from src.utils.config_utils import load_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

logger = setup_logger()


def create_app(cfg) -> FastAPI:
    app = FastAPI()
    app.state.cfg = cfg

    # === CORS ===
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.API.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # === Load model ===
    model = FaceEmotionNet(
        cfg.INFERENCE.YOLO_PATH,
        cfg.INFERENCE.FER_PATH,
        cfg.DISPLAY.VERBOSE
    )
    app.state.model = model

    # === API Routes ===
    app.include_router(router, prefix="/predict", tags=["Predict"])

    # === Serve Frontend ===
    frontend_dir = Path(__file__).parent.parent.parent / "frontend"

    # Serve static JS, CSS, images, etc.
    app.mount("/static", StaticFiles(directory=frontend_dir / "static"), name="static")

    # Optional: Serve env.js if you use dynamic API config
    env_path = frontend_dir / "env.js"
    if env_path.exists():
        @app.get("/env.js")
        def serve_env():
            return FileResponse(env_path)

    # Serve index.html at root
    @app.get("/")
    def serve_index():
        return FileResponse(frontend_dir / "index.html")

    return app

config_path = os.getenv("APP_CONFIG", "src/config/custom.yaml")
cfg = load_config(str(config_path))
app = create_app(cfg)


if __name__ == "__main__":
    logger.info("🚀 Starting FastAPI server in CLI mode...")

    parser = argparse.ArgumentParser(description="Facial Expression Recognition API")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/custom.yaml",
        help="Path to the configuration YAML file"
    )
    parser.add_argument("--host", type=str, help="Host to override the one in config.yaml")
    parser.add_argument("--port", type=int, help="Port to override the one in config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    app = create_app(cfg)

    host = args.host if args.host else cfg.API.HOST
    port = args.port if args.port else cfg.API.PORT

    uvicorn.run(app, host=host, port=port, reload=False)
