"""
Functions run at (or related to) app shutdown
"""

import shutil
from pathlib import Path

from loguru import logger


def app_cleanup():
    """
    Deletes all files and folders in /temp_files/
    """
    logger.info("Deleting temporary app files")
    for item in Path("./temp_files").iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    logger.info("Finished deleting temporary app files")
