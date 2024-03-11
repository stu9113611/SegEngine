from typing import Any
from transformers import SegformerForSemanticSegmentation
import torch
from torch.nn import functional as F


class Segformer(SegformerForSemanticSegmentation):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        pred = super().forward(img)["logits"]
        pred = F.interpolate(pred, img.shape[-2:], mode="bilinear")
        return pred
