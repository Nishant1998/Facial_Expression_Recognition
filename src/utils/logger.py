import logging


def setup_logger():
    """
    Configures logging for the project.

    - Logs INFO level and above to the console.
    - Can be extended to write logs to a file.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    return logging.getLogger(__name__)
