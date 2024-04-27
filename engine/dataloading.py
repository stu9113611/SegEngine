from typing import Any, Optional

from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
import os
import os.path as osp

from engine.category import Category
from engine.transform import Sequence, Transform


class ImgAnnDataset(Dataset):
    def __init__(
        self,
        transforms: list[Transform],
        img_dir: str,
        ann_dir: str,
        max_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.transforms = Sequence(transforms)

        self.img_ann_paths = [
            (
                osp.join(img_dir, fn),
                osp.join(ann_dir, fn.split(".")[0] + ".png"),
            )
            for fn in natsorted(os.listdir(img_dir))
        ]
        if max_len and max_len < len(self.img_ann_paths):
            self.img_ann_paths = self.img_ann_paths[:max_len]

        for _, ann_fn in self.img_ann_paths:
            assert os.path.exists(ann_fn), f"No such file: {ann_fn}"

    def __len__(self) -> int:
        return len(self.img_ann_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, ann_path = self.img_ann_paths[idx]
        return self.transforms({"img_path": img_path, "ann_path": ann_path})

    def get_loader(
        self,
        batch_size: int,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )


class ImgDataset(Dataset):

    def __init__(
        self,
        transforms: list[Transform],
        img_dir: str,
        max_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.transforms = Sequence(transforms)
        self.img_paths = natsorted(
            [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
        )
        if max_len and max_len < len(self.img_paths):
            self.img_paths = self.img_paths[:max_len]

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.transforms(
            {
                "img_path": str(self.img_paths[idx]),
            }
        )

    def get_loader(
        self,
        batch_size: int,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )


if __name__ == "__main__":
    from engine.transform import LoadAnn, LoadImg

    categories = Category.load("data/csv/ceymo.csv")
    dataloader = DataLoader(
        ImgAnnDataset(
            "data/ceymo/clear/train",
            [LoadImg(), LoadAnn(categories)],
            "images",
            "labels",
            ".jpg",
            ".png",
        ),
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    for data in dataloader:
        print(data)
