import os
import random
from glob import glob
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from engine.logger import Logger
from engine.category import Category
from engine.metrics import fast_hist, per_class_iu
import models
import engine.optimizer as optimizer
import engine.transform as transform
from engine.visualizer import IndexMapVisualizer


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_path: str,
        label_path: str,
        transforms: list[transform.Transform],
        max_len: int | None = None,
        image_format: str = ".png",
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.image_format = image_format
        self.fnames = self.get_sorted_filenames(max_len)
        self.transform = transform.Container(transforms)

    def get_sorted_filenames(self, max_len: Optional[int]) -> list[str]:
        filenames = natsorted(
            [Path(fname).stem for fname in glob(os.path.join(self.label_path, "*.png"))]
        )
        if max_len:
            filenames = filenames[:max_len]
        return filenames

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_fname = os.path.join(
            self.image_path, self.fnames[index] + self.image_format
        )
        label_fname = os.path.join(self.label_path, self.fnames[index] + ".png")
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_fname)
        return self.transform(
            {
                "image": image,
                "label": label,
            }
        )

    def get_loader(
        self, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )


class MultiFolderSegmentationDataset(Dataset):
    """
    Every folder must contain images, labels folders.
    Images should be .jpg format, labels should be .png format.
    Filenames of images and label should match.
    """

    def __init__(
        self, folders: list[list[str]], transform: transform.Transform
    ) -> None:
        raise Exception("Not available now! Need to refacter")

        super().__init__()
        self.img_fnames = []
        self.label_fnames = []
        for folder in folders:
            self.img_fnames += glob(folder[0] + "/*.png")
            self.label_fnames += glob(folder[1] + "/*.png")
        self.img_fnames = natsorted(self.img_fnames)
        self.label_fnames = natsorted(self.label_fnames)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_fnames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = cv2.imread(self.img_fnames[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_fnames[index])
        return self.transform(
            {
                "image": image,
                "label": label,
            }
        )


class InferenceImgDataset(Dataset):
    def __init__(
        self,
        image_path: str,
        target_width: int,
        target_height: int,
        max_len: int | None = None,
        image_suffix: str = ".png",
    ) -> None:
        raise Exception("Not available now! Need to refacter")

        super().__init__()
        self.image_path = image_path
        self.fnames = natsorted(glob(os.path.join(self.image_path, "*" + image_suffix)))
        if max_len:
            self.fnames = self.fnames[:max_len]

        self.width = target_width
        self.height = target_height

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> dict:
        image = cv2.imread(self.fnames[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))
        image_tensor = TF.to_tensor(image).view(3, self.height, self.width)
        image_tensor = TF.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return {"image": image_tensor, "background": image, "fname": self.fnames[index]}
