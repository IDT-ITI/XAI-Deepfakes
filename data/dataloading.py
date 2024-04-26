from typing import Any, Optional
import cv2
import numpy as np
import logging as log
from data.lmdb_storage import LMDBFileStorage

from pathlib import Path


class DataReader:
    """Base class for all data readers."""

    def load_image_with_retries(self, path: str, max_retries: int = 10) -> np.ndarray:
        raise NotImplementedError


class FileSystemReader(DataReader):
    """Reader that maps relative paths to absolute paths of the filesystem."""

    def __init__(self, root: Optional[Path]):
        super().__init__()
        assert root is not None, "root must be provided"
        self.root: Path = root

    def load_image_with_retries(self, relative_path: str, max_retries: int = 10) -> np.ndarray:
        """
        Load an image from a path with retries.
        Args:
            path (str): relative path to the image.
            max_retries (int): The maximum number of retries.
        Returns:
            np.array: The image as a numpy array if it was successfully loaded.
        Raises:
            RuntimeError: If the image could not be loaded after max_retries.
        """

        for _ in range(max_retries):
            frame = cv2.imread(str(self.root / relative_path))  
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
        else:
            log.error("Error loading image after {} retries".format(max_retries))
            raise RuntimeError(f"Failed to load image {relative_path} with root {self.root}")


class LMDBFileStorageReader(DataReader):
    """Reader that maps relative paths into an LMDBFileStorage."""

    def __init__(self, storage: LMDBFileStorage):
        super().__init__()
        self.storage: LMDBFileStorage = storage

    def load_image_with_retries(self, path: str, max_retries: int = 10) -> np.ndarray:
        for _ in range(max_retries):
            with self.storage.open_file(path) as stream:
                try:
                    frame = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), 1)  # type: ignore
                except Exception as e:
                    log.error(f"Error loading image {path}: {e}")
                    continue
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame
        else:
            log.error("Error loading image after {} retries".format(max_retries))
            raise RuntimeError(f"Failed to load image {path}")

    def __del__(self):
        self.storage.close()


def create_data_reader(
    lmdb_file_storage_path: Optional[Path] = None, root: Optional[Path] = None
) -> DataReader:
    assert (
        lmdb_file_storage_path is not None or root is not None
    ), "Either lmdb_file_storage_path or root must be provided"

    # Limit the number of OpenCV threads to 2 to utilize multiple processes. Otherwise,
    # each process spawns a number of threads equal to the number of logical cores and
    # the overall performance gets worse due to threads congestion.
    cv2.setNumThreads(2)

    if lmdb_file_storage_path is not None:
        return LMDBFileStorageReader(
            LMDBFileStorage(lmdb_file_storage_path, read_only=True)
        )
    else:
        return FileSystemReader(root=root)
