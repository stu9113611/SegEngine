from typing import Any

from engine.inferencer import BasicInferencer, Inferencer, SlideInferencer
from engine.losses.diceloss import DiceLoss
from rich import print as rprint
from rich import print_json
from segmentation_models_pytorch.losses import FocalLoss
from torch.nn import CrossEntropyLoss, Module


def build_loss_function(cfg: dict[str, Any], show: bool = True) -> Module:
    if show:
        rprint(cfg)

    assert cfg.get("name") in ["cross_entropy_loss", "focal_loss"]

    match cfg.get("name"):
        case "cross_entropy_loss":
            loss_function = CrossEntropyLoss(
                weight=cfg.get("weight"),
                ignore_index=cfg.get("ignore_index"),
                reduction=cfg.get("reduction"),
                label_smoothing=cfg.get("label_smoothing"),
            )

        case "focal_loss":
            loss_function = FocalLoss(
                mode="multiclass",
                alpha=cfg.get("alpha"),
                gamma=cfg.get("gamma"),
                ignore_index=cfg.get("ignore_index"),
                normalized=cfg.get("normalized"),
            )
        case "dice_loss":
            loss_function = DiceLoss()

    return loss_function.to(cfg.get("device"))


def build_inferencer(cfg: dict[str, Any], show: bool = True) -> Inferencer:
    if show:
        rprint(cfg)

    assert cfg.get("mode") in ["basic", "slide"]

    match cfg.get("mode"):
        case "basic":
            inferencer = BasicInferencer()
        case "slide":
            inferencer = SlideInferencer(
                crop_size=cfg["crop_size"],
                stride=cfg["stride"],
                num_categories=cfg["num_categories"],
            )

    return inferencer
