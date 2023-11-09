from __future__ import annotations
import csv
from dataclasses import dataclass
import tabulate

import torch

from engine.core import dict_add

"""
In current version, csv file must contain a ignore category (index = 255)
"""


@dataclass
class Category:
    id: int
    name: str
    r: int
    g: int
    b: int

    @classmethod
    def load(cls, csv_fname: str) -> list[Category]:
        with open(csv_fname, "r", encoding="utf-8-sig") as file:
            cats = [
                Category(int(id), name, int(r), int(g), int(b))
                for id, name, r, g, b in csv.reader(file)
            ]
        return cats

    @classmethod
    def print(cls, categories: list[Category]) -> None:
        print(tabulate.tabulate(categories, headers="keys", tablefmt="grid"))

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, another: Category) -> bool:
        return self.id == another.id


def count_category(
    index_map: torch.Tensor, categories: list[Category]
) -> dict[Category, int]:
    count = {}
    for cat in categories:
        count[cat] = torch.where(index_map == cat.id)[0].shape[0]

    return count


if __name__ == "__main__":
    from engine.dataset import SegmentationDataset
    from engine.transform import ValTransform
    from pprint import pprint
    from tqdm import tqdm
    from collections import Counter

    counts = {}

    categories = Category.load("./csv/gta.csv")
    dataloader = SegmentationDataset(
        "../Cityscapes/train",
        "../Cityscapes/my_train_labels",
        ValTransform(categories, 512, 512),
    ).get_loader(1, False, 8, False)
    for data in tqdm(dataloader):
        image = data["image"].cuda()  # 3 x 512 x 512
        label = data["label"].cuda()  # 512 x 512

        counts = dict_add(counts, count_category(label[0], categories))

    pprint(counts)

    sum = 0
    for cat, count in counts.items():
        sum += count

    for cat, count in counts.items():
        print(f"Category[{cat.name}] count:{count} portion:{count/sum: .4f}")
