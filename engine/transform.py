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
        # cat_ratio: float = 0.0,
        rare_cat_crop: bool = False,
        patient: int = 10,
        efficient: bool = False,
        efficient_interval: int = 10,
    ) -> None:
        self.image_scale = image_scale
        self.scale = scale
        self.crop_size = np.array(crop_size)
        self.antialias = antialias
        self.rare_cat_crop = rare_cat_crop
        self.patient = patient
        self.efficient = efficient
        if self.efficient:
            self.crop_sizes = np.linspace(self.crop_size, self.crop_size // 2, 6)
            self.efficient_interval = efficient_interval
            self.efficient_counter = 0

    def get_random_size(self):
        min_scale, max_scale = self.scale
        random_scale = random.random() * (max_scale - min_scale) + min_scale
        height = int(self.image_scale[0] * random_scale)
        width = int(self.image_scale[1] * random_scale)
        return height, width

    def get_crop_size(self):
        if self.efficient:
            crop_size = self.crop_sizes[self.efficient_counter // 8]
            self.efficient_counter += 1
            if self.efficient_counter == 48:
                self.efficient_counter = 0
        else:
            crop_size = self.crop_size
        return crop_size.astype(int)

    def get_random_crop(self, scaled_height, scaled_width, crop_size):
        crop_y0 = random.randint(0, scaled_height - crop_size[0])
        crop_x0 = random.randint(0, scaled_width - crop_size[1])

        return crop_y0, crop_x0

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        height, width = self.get_random_size()
        crop_size = self.get_crop_size()
        y0, x0 = self.get_random_crop(height, width, crop_size)
        if "ann" in data:
            if self.rare_cat_crop:
                assert (
                    "ann" in data
                ), "Category-ratio cropping is avaliable only when label is given!"
                if "random_cat_id" in data:
                    random_id = data["random_cat_id"]
                else:
                    random_id = random.choice(
                        data["ann"].unique(sorted=False)
                    )  # Choose a random category id in the label
                uncropped_ann = F.resize(
                    data["ann"][:, None, :],
                    (height, width),
                    interpolation=F.InterpolationMode.NEAREST,
                ).squeeze()

                best_ratio = 0
                best_x0 = x0
                best_y0 = y0
                for _ in range(self.patient):
                    ann = uncropped_ann[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                    ratio = (
                        torch.where(ann == random_id)[0].shape[0]
                        / ann.flatten().shape[0]
                    )
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_x0 = x0
                        best_y0 = y0

                    y0, x0 = self.get_random_crop(height, width)

                x0 = best_x0
                y0 = best_y0
                data["ann"] = uncropped_ann[
                    y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]
                ]
            else:
                data["ann"] = F.resize(
                    data["ann"][:, None, :],
                    (height, width),
                    interpolation=F.InterpolationMode.NEAREST,
                ).squeeze()[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

        if "img" in data:
            data["img"] = F.resize(
                data["img"], (height, width), antialias=self.antialias
            )[:, y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

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
