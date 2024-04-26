from typing import Any, Optional, Callable
from data.dataloading import create_data_reader
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import v2
from pathlib import Path


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        lmdb_path: str,
        root: Optional[str] = None,
        transforms: Optional[v2.Compose] = None,
        target_transforms: Optional[Callable] = None,
        task: str = "binary",  # binary or multiclass
    ):
        super().__init__()
        assert Path(lmdb_path).exists(), f"{lmdb_path} does not exist"
        self.lmdb_path = lmdb_path
        assert root == None or Path(root).exists(), f"{root} does not exist"
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.task = task
        self.data_reader = None
        assert task in ["binary", "multiclass"], "task must be binary or multiclass"

        assert Path(csv_path).exists(), f"{csv_path} does not exist"
        self.df = pd.read_csv(csv_path, sep=" ")

        if self.task == "binary":
            self.df = self.df.loc[:, ["relative_path", "bin_label"]]
        elif self.task == "multiclass":
            self.df = self.df.loc[:, ["relative_path", "mc_label"]]

        self.labeled_frame_paths = list(self.df.itertuples(index=False, name=None))

    def __len__(self):
        assert (
            self.labeled_frame_paths is not None
        ), "labeled_frame_paths must be provided"
        return len(self.labeled_frame_paths)

    def __getitem__(self, idx):
        if self.data_reader is None:
            self.data_reader = create_data_reader(self.lmdb_path, None)

        path, label = self.labeled_frame_paths[idx]
        if self.root is not None:
            path = self.root / path
        frame = self.data_reader.load_image_with_retries(str(path), max_retries=10)

        if self.transforms:
            if isinstance(self.transforms, v2.Compose):
                frame = self.transforms(frame)
            else:
                raise NotImplementedError(
                    f"Transforms of type {type(self.transforms)} not implemented."
                )
        if self.target_transforms:
            label = self.target_transforms(label)
        return frame, label
