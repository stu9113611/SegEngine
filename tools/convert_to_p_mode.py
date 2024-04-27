from PIL import Image
import os
import cv2
from tqdm import tqdm
import numpy as np
from functools import partial
import joblib
from multiprocessing import Pool

import argparse
from engine.category import Category
from rich.progress import Progress
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=str)
    parser.add_argument("target_dir", type=str)
    parser.add_argument("category_csv", type=str)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--nproc", type=int, default=8)

    return parser.parse_args()


def convert(
    filenames: str,
    source_dir,
    target_dir,
    palatte: list[Category],
) -> None:
    for filename in filenames:
        image = cv2.imread(os.path.join(source_dir, filename))
        output = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for id, (r, g, b) in enumerate(palatte):
            class_mask = np.equal(image, (b, g, r))
            eq = np.all(class_mask, axis=-1)
            output[eq] = id
        output = Image.fromarray(output).convert("P")
        output.putpalette(palatte)
        output.save(os.path.join(target_dir, filename), bitmap_format=".png")


def main():
    args = parse_args()
    if os.path.exists(args.target_dir):
        print("Target folder already exist!")
        return

    filenames = os.listdir(args.source_dir)
    if len(filenames) == 0:
        print("No label detected.")
        return

    if args.max_len and args.max_len < len(filenames):
        filenames = filenames[: args.max_len]

    os.mkdir(args.target_dir)

    categories = Category.load(args.category_csv)
    palatte = np.array([(cat.r, cat.g, cat.b) for cat in categories], dtype=np.uint8)

    s = math.ceil(len(filenames) / args.nproc)
    splits = [filenames[i * s : (i + 1) * s] for i in range(args.nproc)]

    print("Converting, please wait...")
    joblib.Parallel(n_jobs=args.nproc)(
        [
            joblib.delayed(convert)(split, args.source_dir, args.target_dir, palatte)
            for split in splits
        ]
    )


if __name__ == "__main__":
    main()
