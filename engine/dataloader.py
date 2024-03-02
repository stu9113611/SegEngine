import os
from pathlib import Path
from typing import Any, Optional, Union, Sequence

import cv2
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset

from engine.logger import Logger
from engine.transform import Transform, Composition
from dataclasses import dataclass


class InfiniteDataloader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        drop_last: bool,
        pin_memory: bool,
    ) -> None:
        self.dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self.iterator

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


class ImgAnnDataset(Dataset):
    def __init__(
        self,
        root: str,
        transforms: list[Transform],
        img_prefix: str,
        ann_prefix: str,
        img_suffix: str,
        ann_suffix: str,
        max_len: Optional[int] = None,
        check_exist: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.img_prefix = img_prefix
        self.ann_prefix = ann_prefix
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix

        stems = natsorted(
            [Path(path).stem for path in (self.root / self.img_prefix).iterdir()]
        )
        if max_len:
            assert max_len <= len(
                stems
            ), "Dataset max length must be less than total images"
            stems = stems[:max_len]

        self.img_paths: list[Path] = []
        self.ann_paths: list[Path] = []
        for s in stems:
            self.img_paths.append(
                (self.root / self.img_prefix / s).with_suffix(self.img_suffix)
            )
            self.ann_paths.append(
                (self.root / self.ann_prefix / s).with_suffix(self.ann_suffix)
            )

        if check_exist:
            for img_path, ann_path in zip(self.img_paths, self.ann_paths):
                assert img_path.exists(), "{} doesn't exists.".format(img_path)
                assert ann_path.exists(), "{} doesn't exists.".format(ann_path)

        self.transforms = Composition(transforms)

    def show_info(self, logger: Logger):
        logger.info(
            "ImgAnnDataset",
            "Image path: {}".format(self.root / self.img_prefix),
        )
        logger.info(
            "ImgAnnDataset",
            "Annotation path: {}".format(self.root / self.ann_prefix),
        )
        logger.info("ImgAnnDataset", "Image suffix: {}".format(self.img_suffix))
        logger.info("ImgAnnDataset", "Annotation suffix: {}".format(self.ann_suffix))
        logger.info("ImgAnnDataset", "Data found: {}".format(self.__len__()))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.transforms.transform(
            {
                "img_path": str(self.img_paths[idx]),
                "ann_path": str(self.ann_paths[idx]),
            }
        )

    def get_loader(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        drop_last: bool,
        pin_memory: bool,
        infinite: bool = False,
    ) -> DataLoader:
        if infinite:
            return InfiniteDataloader(
                self,
                batch_size,
                shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=pin_memory,
            )
        else:
            return DataLoader(
                self,
                batch_size,
                shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=pin_memory,
            )
