import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from rich import print
from rich.progress import Progress
from rich.table import Table

import engine.transform as transform
from engine.category import Category
from engine.dataloading import ImgAnnDataset, ImgDataset
from engine.inference import Inferencer, SlideInferencer
from engine.metric import Metrics
from engine.model.segformer import Segformer
from engine.visualizer import IdMapVisualizer, ImgSaver


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("img_dir", type=str)
    parser.add_argument("ann_dir", type=str)
    parser.add_argument("category_csv", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=None)
    args = parser.parse_args()
    return args


def main(args: Namespace):

    image_size = 1080, 1920  # We use HxW in this entire project.
    crop_size = 512, 512
    stride = 384, 384

    device = "cuda" if torch.cuda.is_available() else "cpu"

    categories = Category.load(args.category_csv)
    num_categories = Category.get_num_categories(
        categories
    )  # does not include ignored categories

    model = Segformer.from_pretrained("nvidia/mit-b0", num_labels=num_categories).to(
        device
    )
    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model.eval()

    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        transform.Resize(image_size),
        transform.Normalize(),
    ]

    inferencer = SlideInferencer(crop_size, stride, num_categories)
    if args.save_dir:
        visualizer = IdMapVisualizer(categories)
        img_saver = ImgSaver(args.save_dir, visualizer)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    metrics = Metrics(num_categories, nan_to_num=0)

    dataloader = ImgAnnDataset(
        transforms=transforms,
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        max_len=args.max_len,
    ).get_loader(
        batch_size=args.batch_size, pin_memory=False, num_workers=args.num_workers
    )

    with Progress() as prog:
        with torch.no_grad():
            task = prog.add_task("Test", total=len(dataloader))
            avg_loss = 0
            for data in dataloader:
                img = data["img"].to(device)
                ann = data["ann"].to(device)[:, 0, :, :]

                pred = inferencer.inference(model, img)
                avg_loss += criterion(pred, ann).item()
                metrics.compute_and_accum(pred.argmax(1), ann)

                if args.save_dir:
                    for fn, p in zip(data["img_path"], pred):
                        img_saver.save_pred(p[None, :], Path(fn).stem + ".png")

                prog.update(task, advance=1)

            result = metrics.get_and_reset()

            table = Table()
            table.add_column("Category")
            table.add_column("Acc")
            table.add_column("IoU")
            table.add_column("Dice")
            table.add_column("Fscore")
            table.add_column("Precision")
            table.add_column("Recall")
            for cat, acc, iou, dice, fs, pre, rec in zip(
                categories,
                result["Acc"],
                result["IoU"],
                result["Dice"],
                result["Fscore"],
                result["Precision"],
                result["Recall"],
            ):
                table.add_row(
                    cat.name,
                    "{:.5f}".format(acc),
                    "{:.5f}".format(iou),
                    "{:.5f}".format(dice),
                    "{:.5f}".format(fs),
                    "{:.5f}".format(pre),
                    "{:.5f}".format(rec),
                )
            table.add_row(
                "Avg.",
                "{:.5f}".format(result["Acc"].mean()),
                "{:.5f}".format(result["IoU"].mean()),
                "{:.5f}".format(result["Dice"].mean()),
                "{:.5f}".format(result["Fscore"].mean()),
                "{:.5f}".format(result["Precision"].mean()),
                "{:.5f}".format(result["Recall"].mean()),
            )
            avg_loss /= len(dataloader)

            prog.remove_task(task)

            print(table)
            print("Average Loss:", avg_loss)


if __name__ == "__main__":
    args = parse_args()
    main(args)
