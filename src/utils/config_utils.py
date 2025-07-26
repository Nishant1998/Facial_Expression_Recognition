import os
from src.config import _C
from src.utils.logger import setup_logger

logger = setup_logger()

def get_cfg_defaults():
    """Get a copy of the default config."""
    return _C.clone()

def load_config(custom_config_path="config/custom.yaml"):
    """
    Load default configuration and override with custom YAML if available.

    :param custom_config_path: Path to the custom YAML configuration file.
    :return: Loaded configuration object.
    """
    logger.info("Loading configuration...")

    cfg = get_cfg_defaults()

    if os.path.exists(custom_config_path):
        try:
            cfg.merge_from_file(custom_config_path)
            logger.info(f"Successfully loaded custom configuration from {custom_config_path}")
        except Exception as e:
            logger.error(f"Error loading custom configuration: {e}", exc_info=True)
            logger.warning("Falling back to default configuration.")
    else:
        logger.warning(f"Custom config {custom_config_path} not found. Using default settings.")

    logger.info("Configuration loaded successfully.")
    return cfg