import argparse
import os
import sys

import numpy as np
import torch
from torchvision.datasets import MNIST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the MNIST dataset to a plaintext file where each line is:\n"
            "label pixel0 pixel1 ... pixel783 (integers in [0,255])."
        )
    )
    parser.add_argument("--output", default = "mnist_raw.txt", help="Destination text file (will be overwritten)")
    parser.add_argument("--root", default="./data", help="Root directory for torchvision MNIST download/cache")
    parser.add_argument("--train", action="store_true", default = True, help="Use the training set (default: test set)")
    parser.add_argument("--limit", type=int, default=None, help="Only convert the first N images (for quick tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = MNIST(root=args.root, train=args.train, download=True)
    total = len(dataset) if args.limit is None else min(len(dataset), args.limit)

    split = "train" if args.train else "test"
    print(f"Writing {total} {split} images to {args.output}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, (img, label) in enumerate(dataset):
            if idx >= total:
                break

            # Convert PIL Image (28x28 grayscale) to flat list of uint8 ints
            pixels = np.array(img, dtype=np.uint8).reshape(-1)
            line = " ".join([str(label)] + [str(p) for p in pixels])
            out_f.write(line + "\n")

            if (idx + 1) % 1000 == 0 or idx + 1 == total:
                print(f"{idx + 1}/{total} written", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
