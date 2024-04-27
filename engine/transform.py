import random
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as t
from torchvision.io import read_image
from torchvision.transforms import functional as tf


class Transform(ABC):
    @abstractmethod
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]: ...


class LoadImg(Transform):
    def __init__(self, mode: str = "rgb") -> None:
        assert mode in ["rgb", "bgr"]
        self.to_bgr = mode == "bgr"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "img_path" in data, "No path given to load image!"

        img = read_image(data["img_path"])
        if self.to_bgr:
            img = img[[2, 1, 0], :]
        data["img"] = img.float() / 255.0
        return data


class LoadAnn(Transform):
    def __init__(self, ignored_index: Optional[list[int]] = None) -> None:
        self.ignored_index = ignored_index

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "ann_path" in data, "No path given to load annotation!"

        img = Image.open(data["ann_path"])
        data["ann"] = tf.pil_to_tensor(img).long()
        return data


class Resize(Transform):
    def __init__(self, size: tuple[int, int]) -> None:
        self.img_transform = t.Resize(size, antialias=True)
        self.ann_transform = t.Resize(
            size, interpolation=tf.InterpolationMode.NEAREST, antialias=True
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = self.img_transform(data["img"])
        if "ann" in data:
            data["ann"] = self.ann_transform(data["ann"])
        return data


class RandomResizeCrop:
    def __init__(
        self,
        image_scale: tuple[int, int],
        scale: tuple[float, float],
        crop_size: tuple[int, int],
        antialias: bool = True,
    ) -> None:
        self.image_scale = image_scale
        self.scale = scale
        self.crop_size = np.array(crop_size)
        self.antialias = antialias

    def get_random_size(self):
        min_scale, max_scale = self.scale
        random_scale = random.random() * (max_scale - min_scale) + min_scale
        height = int(self.image_scale[0] * random_scale)
        width = int(self.image_scale[1] * random_scale)
        return height, width

    def get_random_crop(self, scaled_height, scaled_width, crop_size):
        crop_y0 = random.randint(0, scaled_height - crop_size[0])
        crop_x0 = random.randint(0, scaled_width - crop_size[1])

        return crop_y0, crop_x0

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        height, width = self.get_random_size()
        y0, x0 = self.get_random_crop(height, width, self.crop_size)

        if "img" in data:
            data["img"] = tf.resize(
                data["img"], (height, width), antialias=self.antialias
            )[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        if "ann" in data:
            data["ann"] = tf.resize(
                data["ann"],
                (height, width),
                interpolation=tf.InterpolationMode.NEAREST,
            )[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        return data


class RandomHorizontalFlip(Transform):
    def __init__(self) -> None:
        self.flip = t.RandomHorizontalFlip()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = self.flip(data["img"])
        if "ann" in data:
            data["ann"] = self.flip(data["ann"])
        return data


class ColorJitter:
    def __init__(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
    ) -> None:
        self.jitter = t.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = self.jitter(data["img"])

        return data


class Normalize(Transform):
    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.normalize = t.Normalize(mean, std, True)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "img" in data:
            data["img"] = self.normalize(data["img"])
        return data


class Sequence(Transform):
    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data


if __name__ == "__main__":
    from torchvision.io import write_jpeg

    transforms = [LoadImg(), Resize(512, 512)]

    data = {
        "filename": "./data/rlmd/clear/images/train/10.jpg",
    }

    for transform in transforms:
        data = transform(data)

    write_jpeg(data["img"], "test.png")
