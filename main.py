import argparse

from src.serve.inferencer import infer
from src.train.trainer import train_model
from src.utils import setup_logger
from src.utils.config_utils import load_config

logger = setup_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Expression Recognition")
    parser.add_argument("--config", type=str, default="src/config/custom.yaml", help="Path to the configuration YAML file")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    if cfg.IS_TRAINING:
        train_model(cfg)
    else:
        infer(cfg)


