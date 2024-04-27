import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from rich import print
from rich.progress import Progress

import engine.transform as transform
from engine.category import Category
from engine.dataloading import ImgDataset
from engine.inference import Inferencer, SlideInferencer
from engine.visualizer import IdMapVisualizer, ImgSaver
from engine.model.segformer import Segformer


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("category_csv", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=None)
    args = parser.parse_args()
    return args


def inference_image(
    filename: str,
    transforms: list[transform.Transform],
    model: torch.nn.Module,
    inferencer: Inferencer,
    img_saver: ImgSaver,
    device: str,
):
    data = {"filename": filename}

    for trans in transforms:
        data = trans(data)
        img = data["img"].to(device)

    pred = inferencer.inference(model, img[None, :])

    img_saver.save_pred(pred, Path(data["filename"]).stem + ".png")


def inference_dir(
    img_path: str,
    batch_size: int,
    num_workers: int,
    max_len: int,
    transforms: list[transform.Transform],
    model: torch.nn.Module,
    inferencer: Inferencer,
    img_saver: ImgSaver,
    device: str,
):
    print("Inference a folder.")
    print(f"Batch size: {batch_size}, Number workers: {num_workers}")

    dataloader = ImgDataset(
        transforms=transforms, img_dir=img_path, max_len=max_len
    ).get_loader(batch_size=batch_size, pin_memory=False, num_workers=num_workers)

    with Progress() as prog:
        task = prog.add_task("Inference", total=len(dataloader))
        for data in dataloader:
            img = data["img"].to(device)

            pred = inferencer.inference(model, img)

            for path, p in zip(data["img_path"], pred):
                img_saver.save_pred(p[None, :], Path(path).stem + ".png")

            prog.update(task, advance=1)
        prog.remove_task(task)


def main(args: Namespace):

    image_size = 1080, 1920  # We use HxW in this entire project.
    crop_size = 512, 512
    stride = 384, 384

    device = "cuda" if torch.cuda.is_available() else "cpu"

    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(
        categories
    )  # does not include ignored categories

    model = Segformer.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_categories,
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model.eval()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    transforms = [
        transform.LoadImg(),
        transform.Resize(image_size),
        transform.Normalize(),
    ]

    inferencer = SlideInferencer(crop_size, stride, num_categories)
    visualizer = IdMapVisualizer(categories)
    img_saver = ImgSaver(args.save_path, visualizer)

    with torch.no_grad():
        if os.path.isfile(args.img_path):
            inference_image(
                args.img_path,
                transforms,
                model,
                inferencer,
                img_saver,
                device,
            )

        else:
            inference_dir(
                args.img_path,
                args.batch_size,
                args.num_workers,
                args.max_len,
                transforms,
                model,
                inferencer,
                img_saver,
                device,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
