import os
from pathlib import Path
from typing import Any, Optional, Union, Sequence

import cv2
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
import torch

from engine.logger import Logger
from engine.transform import Transform, Composition
from dataclasses import dataclass
import random
import numpy as np
from engine.category import Category
from rich.table import Table
import json
from rich import print as rprint
from copy import deepcopy


@dataclass
class RCSConfig:
    file_path: str
    ignore_ids: list[int]
    temperature: float = 0.1


class RareCategoryManager:
    def __init__(
        self,
        categories: list[Category],
        rcs_cfg: RCSConfig,
        show: bool = True,
    ) -> None:
        with open(rcs_cfg.file_path) as f:
            data = json.load(f)

        self.stems = {cat.id: [] for cat in categories}
        category_probs = torch.zeros(len(categories))
        ignore = [True if cat.id in rcs_cfg.ignore_ids else False for cat in categories]
        for d in data:
            filename = Path(d["filename"]).stem
            count = np.array(d["count"])
            for cat in categories:
                if count[cat.id]:
                    self.stems[cat.id].append(filename)
            category_probs += count
        self.consumable_stems = deepcopy(self.stems)
        category_probs[ignore] = 0
        category_probs /= category_probs.sum()
        self.sampling_probs = ((1 - category_probs) / rcs_cfg.temperature).exp()
        self.sampling_probs /= self.sampling_probs[category_probs != 0].sum()
        self.sampling_probs[category_probs == 0] = 0

        self.length = len(data)

        if show:
            table = Table("Category Name", "Category Prob.", "Sampling Prob.")
            for cat, cprob, sprob in zip(
                categories, category_probs, self.sampling_probs
            ):
                if cat.id in rcs_cfg.ignore_ids:
                    table.add_row(
                        cat.name,
                        "ignored",
                        "{:.5f}".format(sprob.item()),
                        style="cyan",
                    )
                else:
                    table.add_row(
                        cat.name,
                        "{:.5f}".format(cprob.item()),
                        "{:.5f}".format(sprob.item()),
                    )
            rprint(table)

    def get_rare_cat_id(self) -> int:
        return np.random.choice(
            [i for i in range(len(self.sampling_probs))],
            replace=True,
            p=self.sampling_probs.numpy(),
        )

    def get_stems(self, i: int) -> list[Path]:
        if len(self.consumable_stems[i]) == 0:
            # print(f"Already trained all images in category {i}!")
            self.consumable_stems[i] = deepcopy(self.stems[i])
        return self.consumable_stems[i]


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


class RCSImgAnnDataset(Dataset):
    def __init__(
        self,
        root: str,
        transforms: list[Transform],
        img_prefix: str,
        ann_prefix: str,
        img_suffix: str,
        ann_suffix: str,
        categories: list[Category],
        rcs_cfg: RCSConfig,
        show_rcs: bool = True,
        # check_exist: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.img_prefix = img_prefix
        self.ann_prefix = ann_prefix
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix

        self.rcm = RareCategoryManager(categories, rcs_cfg, show_rcs)
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
        # return len(self.img_paths)
        return self.rcm.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # we don't use the index given by the dataloader
        random_cat_id = self.rcm.get_rare_cat_id()
        stems = self.rcm.get_stems(random_cat_id)
        stem = random.choice(stems)
        stems.remove(stem)
        return self.transforms.transform(
            {
                "img_path": str(
                    (self.root / self.img_prefix / stem).with_suffix(self.img_suffix)
                ),
                "ann_path": str(
                    (self.root / self.ann_prefix / stem).with_suffix(self.ann_suffix)
                ),
                "random_cat_id": random_cat_id,
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


if __name__ == "__main__":
    import json
    from rich import print as rprint
    from engine.transform import LoadImg, LoadAnn

    categories = Category.load("data/csv/vpgnet.csv", False)
    dataloader = RCSImgAnnDataset(
        "data/vpgnet/clear/train",
        [LoadImg(), LoadAnn(categories)],
        "images",
        "labels",
        ".png",
        ".png",
        categories=categories,
        rcs_path="vpgnet.json",
        ignore_ids=[0],
    ).get_loader(4, False, 4, True, False, True)

    for data in dataloader:
        print(data)
