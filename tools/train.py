import os
from argparse import ArgumentParser, Namespace

import torch
from rich import print
from rich.progress import Progress
from rich.table import Table

import engine.transform as transform
from engine.category import Category
from engine.dataloading import ImgAnnDataset
from engine.inference import SlideInferencer
from engine.logger import Logger
from engine.metric import Metrics
from engine.model.segformer import Segformer
from engine.visualizer import IdMapVisualizer, ImgSaver


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("train_img_dir", type=str)
    parser.add_argument("train_ann_dir", type=str)
    parser.add_argument("val_img_dir", type=str)
    parser.add_argument("val_ann_dir", type=str)
    parser.add_argument("category_csv", type=str)
    parser.add_argument("max_epochs", type=int)
    parser.add_argument("logdir", type=str)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-max-len", type=int, default=None)
    parser.add_argument("--val-max-len", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--resume", type=int, default=False)
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

    model = Segformer.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_categories,
    ).to(device)

    train_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        transform.RandomResizeCrop(image_size, (0.5, 2), crop_size),
        transform.ColorJitter(0.5, 0.5, 0.5),
        transform.Normalize(),
    ]

    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        transform.Resize(image_size),
        transform.Normalize(),
    ]

    inferencer = SlideInferencer(crop_size, stride, num_categories)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    metrics = Metrics(num_categories, nan_to_num=0)

    train_dataloader = ImgAnnDataset(
        transforms=train_transforms,
        img_dir=args.train_img_dir,
        ann_dir=args.train_ann_dir,
        max_len=args.train_max_len,
    ).get_loader(
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    val_dataloader = ImgAnnDataset(
        transforms=val_transforms,
        img_dir=args.val_img_dir,
        ann_dir=args.val_ann_dir,
        max_len=args.val_max_len,
    ).get_loader(
        batch_size=1,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.segformer.parameters(), "lr": 6e-5},
            {"params": model.decode_head.parameters(), "lr": 6e-4},
        ]
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 1e-4, 1, len(train_dataloader)
    )
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, args.max_epochs, 1
    )

    if args.resume:
        checkpoint = torch.load(
            os.path.join(args.logdir, f"checkpoint_{args.resume}.pth")
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler"])
        poly_scheduler.load_state_dict(checkpoint["poly_scheduler"])
        start_epoch = args.resume + 1
    else:
        start_epoch = 1

    if not args.resume and os.path.exists(args.logdir):
        raise FileExistsError(
            f"{args.logdir} already exists. Please specify a different logdir or resume a checkpoint in this logdir."
        )
    logger = Logger(args.logdir)
    img_saver = ImgSaver(args.logdir, IdMapVisualizer(categories))

    with Progress() as prog:

        whole_task = prog.add_task("Training", total=args.max_epochs)
        for e in range(start_epoch, args.max_epochs + 1):
            train_task = prog.add_task(f"Train - {e}", total=len(train_dataloader))
            train_avg_loss = 0
            model.train()
            for data in train_dataloader:
                img = data["img"].to(device)
                ann = data["ann"].to(device)[:, 0, :, :]

                optimizer.zero_grad()

                pred = model(img)
                loss = criterion(pred, ann)
                loss.backward()

                optimizer.step()

                train_avg_loss += loss.item()

                warmup_scheduler.step()
                prog.update(train_task, advance=1)

            train_avg_loss /= len(train_dataloader)

            logger.info("TrainLoop", f"Loss: {train_avg_loss:.5f}")
            logger.tb_log("TrainLoop/Loss", train_avg_loss, e)

            if e % args.save_interval == 0:
                img_saver.save_img(img, f"train_{e}_img.png")
                img_saver.save_ann(ann, f"train_{e}_ann.png")
                img_saver.save_pred(pred, f"train_{e}_pred.png")

            prog.remove_task(train_task)

            if e % args.val_interval == 0:
                with torch.no_grad():
                    val_task = prog.add_task(f"Val - {e}", total=len(val_dataloader))
                    val_avg_loss = 0
                    model.eval()
                    for data in val_dataloader:
                        img = data["img"].to(device)
                        ann = data["ann"].to(device)[:, 0, :, :]

                        pred = inferencer.inference(model, img)
                        val_avg_loss += criterion(pred, ann).item()
                        metrics.compute_and_accum(pred.argmax(1), ann)

                        prog.update(val_task, advance=1)

                    if args.logdir:
                        img_saver.save_img(img, f"val_{e}_img.png")
                        img_saver.save_ann(ann, f"val_{e}_ann.png")
                        img_saver.save_pred(pred, f"val_{e}_pred.png")

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
                    val_avg_loss /= len(val_dataloader)

                    prog.remove_task(val_task)

                    print(table)
                    logger.info("ValLoop", f"Loss: {val_avg_loss:.5f}")
                    logger.tb_log("ValLoop/Loss", val_avg_loss, e)
                    logger.tb_log("ValLoop/mIoU", result["IoU"].mean(), e)

            poly_scheduler.step()
            if e % args.checkpoint_interval == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "warmup_scheduler": warmup_scheduler.state_dict(),
                        "poly_scheduler": poly_scheduler.state_dict(),
                    },
                    os.path.join(args.logdir, f"checkpoint_{e}.pth"),
                )
            prog.update(whole_task, advance=1)
        prog.remove_task(whole_task)


if __name__ == "__main__":
    args = parse_args()
    main(args)
