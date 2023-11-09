from typing import Any, Protocol
import torch
from category import Category
import random
from numpy.typing import NDArray
import numpy as np
import cv2
import math
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter, InterpolationMode
from skimage.util import random_noise


class Transform(Protocol):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        ...


def rotate(
    image: NDArray,
    angle: float,
    center: tuple[float, float] | None = None,
    scale: float = 1.0,
):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def get_index_map(label: NDArray, categories: list[Category]) -> torch.Tensor:
    index_map = np.zeros((label.shape[0], label.shape[1]), dtype=float)
    for cat in categories:
        eq = np.where(
            (label[:, :, 0] == cat.b)
            & (label[:, :, 1] == cat.g)
            & (label[:, :, 2] == cat.r)
        )
        index_map[eq] = cat.id
    return torch.from_numpy(index_map).long()


class Scale:
    def __init__(self, target_size: tuple[int, int]) -> None:
        self.size = target_size

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"]
        label = data["label"]

        image = TF.resize(image, self.size, InterpolationMode.BILINEAR, antialias=False)
        label = TF.resize(label, self.size, antialias=False).numpy().transpose(1, 2, 0)

        return {"image": image, "label": label}


class InboundRandomScaledRotation:
    def __init__(self, target_size: tuple[int, int]) -> None:
        self.width = target_size[0]
        self.height = target_size[1]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"]
        label = data["label"]

        image = TF.resize(image, (self.height, self.width), antialias=False)
        label = TF.resize(
            label, (self.height, self.width), InterpolationMode.NEAREST, antialias=False
        )

        rotate_angle = random.randint(-30, 30)
        image = TF.rotate(image, rotate_angle, expand=True)
        label = TF.rotate(label, rotate_angle, expand=True, fill=[255, 0, 255])

        ratio = abs(
            math.sin(math.radians(rotate_angle)) * math.cos(math.radians(rotate_angle))
        )

        crop_width = int(image.shape[2] / (2 * ratio + 1))
        crop_height = int(image.shape[1] / (2 * ratio + 1))
        left = int(crop_height * ratio)
        top = int(crop_width * ratio)

        image = TF.resized_crop(
            image,
            top,
            left,
            crop_height,
            crop_width,
            (self.height, self.width),
            antialias=False,
        )
        label = (
            TF.resized_crop(
                label,
                top,
                left,
                crop_height,
                crop_width,
                (self.height, self.width),
                InterpolationMode.NEAREST,
                antialias=False,
            )
            .numpy()
            .transpose(1, 2, 0)
        )

        return {"image": image, "label": label}


class OutboundRandomScaledRotation:
    def __init__(self, target_size: tuple[int, int]) -> None:
        self.width = target_size[0]
        self.height = target_size[1]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"]
        label = data["label"]

        scale = random.random() + 1
        scaled_w = int(self.width * scale)
        scaled_h = int(self.height * scale)
        crop_x = random.randint(0, scaled_w - self.width)
        crop_y = random.randint(0, scaled_h - self.height)
        rotate_angle = random.randint(-30, 30)

        image = TF.resize(image, (scaled_h, scaled_w), antialias=False)
        image = TF.crop(image, crop_y, crop_x, self.height, self.width)
        image = TF.rotate(image, rotate_angle)

        label = TF.resize(
            label, (scaled_h, scaled_w), InterpolationMode.NEAREST, antialias=False
        )
        label = TF.crop(label, crop_y, crop_x, self.height, self.width)
        label = (
            TF.rotate(label, rotate_angle, fill=[255, 0, 255])
            .numpy()
            .transpose(1, 2, 0)
        )

        return {"image": image, "label": label}


class Normalize:
    def __init__(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = TF.normalize(
            data["image"], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return {"image": image, "label": data["label"]}


class ToIndexMap:
    def __init__(self, categories: list[Category]) -> None:
        self.categories = categories

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        label = get_index_map(data["label"], self.categories)
        return {"image": data["image"], "label": label}


class Container:
    def __init__(self, transforms: list[Transform]) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)

        return data
