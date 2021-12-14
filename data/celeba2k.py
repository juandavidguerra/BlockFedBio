from collections import namedtuple
import csv
from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Tuple
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg

CSV = namedtuple("CSV", ["header", "index", "data"])

class CelebA(VisionDataset):
    base_folder = "CelebA-2k"
    def __init__(
            self,
            root: str,
            split: str = "train",            
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,) -> None: 
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split        
        split_map = {
            "train": 0,
            "valid": 1,            
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",("train", "valid", "all"))]
        splits = self._load_csv("list_eval_partition2k.csv")
        identity = self._load_csv("identity_celeba2k.csv")
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        self.filename = splits.index
        self.identity = identity.data[mask]

    def _load_csv(
        self,
        filename: str,
    ) -> CSV:
        data, indices, headers = [], [], []
        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=',', skipinitialspace=True))
        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        target: Any = []
        target.append(self.identity[index, 0])
        if self.transform is not None:
            X = self.transform(X)
        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.identity)