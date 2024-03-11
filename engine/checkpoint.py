from dataclasses import dataclass
import torch


@dataclass
class Checkpoint:
    epochs: int
    state_dict: torch.nn.Module

    def save(self, path: str) -> None:
        torch.save({"epochs": self.epochs, "state_dict": self.state_dict}, path)

    def load(self, path: str) -> None:
        f = torch.load(path)
        self.epochs = f["epochs"]
        self.state_dict = f["state_dict"]
