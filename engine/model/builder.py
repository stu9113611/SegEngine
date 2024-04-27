from typing import Any
import torch
from rich import print
from engine.model.segformer import Segformer


def build_model(config: dict[str, Any], show: bool = True) -> torch.nn.Module:
    if show:
        print(config)

    model_name = config.get("name")
    match model_name:
        case "segformer":
            model = Segformer.from_pretrained(
                pretrained_model_name_or_path=config.get("pretrained"),
                num_labels=config.get("num_categories"),
            )
        case _:
            print(f"Model name not found: {model_name}")
            raise NotImplementedError()

    return model
