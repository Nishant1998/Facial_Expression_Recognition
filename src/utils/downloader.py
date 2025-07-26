import os
import urllib.request
from urllib.error import URLError, HTTPError, ContentTooShortError
from pathlib import Path
from tqdm import tqdm
from src.utils import setup_logger

logger = setup_logger()


def download_weights(path: str, url: str) -> None:
    """
    Downloads a file from the given URL to the specified path, showing a tqdm progress bar.

    :param path: Target file path to save the downloaded weights.
    :param url: URL to download the weights from.
    """
    try:
        path = Path(path)
        if path.exists():
            logger.info(f"Model weights already exist at '{path}'. Skipping download.")
            return

        if not path.parent.exists():
            logger.info(f"Creating directory '{path.parent}'")
            os.makedirs(path.parent, exist_ok=True)

        logger.info(f"Starting download from {url} to '{path}'")

        # Progress bar function
        def show_progress_bar(block_num, block_size, total_size):
            if show_progress_bar.pbar is None:
                show_progress_bar.pbar = tqdm(total=total_size, unit='B', unit_scale=True,
                                              unit_divisor=1024, desc=path.name)
            downloaded = block_num * block_size
            show_progress_bar.pbar.update(downloaded - show_progress_bar.pbar.n)

        show_progress_bar.pbar = None

        urllib.request.urlretrieve(url, str(path), reporthook=show_progress_bar)
        if show_progress_bar.pbar:
            show_progress_bar.pbar.close()

        logger.info(f"Download completed successfully: {path}")

    except HTTPError as e:
        logger.error(f"HTTP Error {e.code} while downloading {url}", exc_info=True)
    except URLError as e:
        logger.error(f"URL Error: {e.reason}", exc_info=True)
    except ContentTooShortError as e:
        logger.error("Download failed: content too short.", exc_info=True)
    except PermissionError as e:
        logger.error(f"Permission denied: Cannot write to '{path}'", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error while downloading model: {e}", exc_info=True)
