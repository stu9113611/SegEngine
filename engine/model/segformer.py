from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch
from torch.nn import functional as F


class Segformer(SegformerForSemanticSegmentation):
    """
    Example:
        device = "cuda"
        model = Segformer.from_pretrained("nvidia/mit-b0", num_labels=25).to(device)

    """

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        pred = super().forward(img).logits
        pred = F.interpolate(pred, img.shape[-2:], mode="bilinear")
        return pred
