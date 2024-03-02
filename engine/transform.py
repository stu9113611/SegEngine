from typing import Any, Protocol, Optional
from torchvision.transforms import functional as F
from engine.category import Category
import cv2
import torch
import numpy as np
import math
import random


class Transform(Protocol):
    def transform(self, data: dict[str, Any]) -> dict[str, Any]: ...


class Composition:
    def __init__(self, transformations: list[Transform]) -> None:
        self.transformations = transformations

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        for transformation in self.transformations:
            data = transformation.transform(data)
        return data


class LoadImg:
    def __init__(self, to_rgb: bool = True) -> None:
        self.to_rgb = to_rgb

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        image = cv2.imread(data["img_path"])
        if self.to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data["img"] = F.to_tensor(image)
        return data


class LoadAnn:
    def __init__(self, categories: list[Category]) -> None:
        self.categories = categories

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        ann = cv2.imread(data["ann_path"])  # bgr

        id_map = np.zeros(ann.shape[:2])
        for cat in self.categories:
            id_map[
                np.where(
                    (ann[:, :, 0] == cat.b)
                    & (ann[:, :, 1] == cat.g)
                    & (ann[:, :, 2] == cat.r)
                )
            ] = cat.id
        data["ann"] = torch.from_numpy(id_map)[None, :].long()
        return data


class Resize:
    def __init__(
        self, image_scale: Optional[tuple[int, int]] = None, antialias: bool = True
    ) -> None:
        self.image_scale = image_scale
        self.antialias = antialias

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["img"].shape[-2:]
            )

            data["img"] = F.resize(data["img"], _image_scale, antialias=self.antialias)

        if "ann" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["ann"].shape[-2:]
            )
            data["ann"] = F.resize(
                data["ann"][:, None, :],
                _image_scale,
                interpolation=F.InterpolationMode.NEAREST,
            ).squeeze()

        return data


class RandomResizeCrop:
    def __init__(
        self,
        image_scale: tuple[int, int],
        scale: tuple[float, float],
        crop_size: tuple[int, int],
        antialias: bool = True,
        cat_ratio: float = 0.0,
        patient: int = 10,
    ) -> None:
        self.image_scale = image_scale
        self.scale = scale
        self.crop_size = crop_size
        self.antialias = antialias
        self.cat_ratio = cat_ratio
        self.patient = patient

    def get_random_size(self):
        min_scale, max_scale = self.scale
        random_scale = random.random() * (max_scale - min_scale) + min_scale
        height = int(self.image_scale[0] * random_scale)
        width = int(self.image_scale[1] * random_scale)
        return height, width

    def get_random_crop(self, scaled_height, scaled_width):
        crop_y0 = random.randint(0, scaled_height - self.crop_size[0])
        crop_x0 = random.randint(0, scaled_width - self.crop_size[1])

        return crop_y0, crop_x0

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        height, width = self.get_random_size()
        y0, x0 = self.get_random_crop(height, width)
        if "ann" in data:
            if self.cat_ratio > 0.0:
                assert (
                    "ann" in data
                ), "Category-ratio cropping is avaliable only when label is given!"
                random_id = random.choice(
                    data["ann"].unique(sorted=False)
                )  # Choose a random category id in the label
                uncropped_ann = F.resize(
                    data["ann"][:, None, :],
                    (height, width),
                    interpolation=F.InterpolationMode.NEAREST,
                ).squeeze()

                for _ in range(self.patient):
                    ann = uncropped_ann[
                        y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
                    ]

                    if (
                        torch.where(ann == random_id)[0].shape[0]
                        / ann.flatten().shape[0]
                        >= self.cat_ratio
                    ):

                        break

                    y0, x0 = self.get_random_crop(height, width)

                data["ann"] = ann
            else:
                data["ann"] = F.resize(
                    data["ann"][:, None, :],
                    (height, width),
                    interpolation=F.InterpolationMode.NEAREST,
                ).squeeze()[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        if "img" in data:
            data["img"] = F.resize(
                data["img"], (height, width), antialias=self.antialias
            )[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        return data


class Normalize:
    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = mean
        self.std = std

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = F.normalize(data["img"], self.mean, self.std)

        return data
