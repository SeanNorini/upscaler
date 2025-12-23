import argparse
import os
import time
import cv2 as cv
import numpy as np

from image_enhancement.scunet import denoise
from image_enhancement.upscaler import upscale


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image enhancer")
    parser.add_argument(
        "-i", "--input", default="./images", help="Path to input image or folder"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./output",
        help="Where to save processed image",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=2048,
        help="Target output size for upscaling (longest side).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="Real_ESRGAN_Video_4x",
        help="Upscaling model to use",
    )
    parser.add_argument(
        "--mode",
        choices=["upscale", "denoise"],
        default="upscale",
        help="Processing mode",
    )
    parser.add_argument(
        "--strength",
        type=int,
        choices=[15, 25, 50],
        default=15,
        help="Denoising strength (only used in denoise mode)",
    )
    return parser.parse_args()


def get_files(path: str) -> tuple[str, list[str]]:
    if os.path.isdir(path):
        return path, os.listdir(path)
    return os.path.dirname(path), [os.path.basename(path)]


def process_image(img: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.mode == "upscale":
        return upscale(img, args.model, args.size)

    if args.mode == "denoise":
        return denoise(img, strength=args.strength)

    raise ValueError(f"Unsupported mode: {args.mode}")


def main() -> None:
    args = get_args()
    folder_path, files = get_files(args.input)

    os.makedirs(args.output, exist_ok=True)

    total: int = len(files)
    batch_start: float = time.time()

    for idx, file in enumerate(files, start=1):
        file_start: float = time.time()

        img = cv.imread(
            os.path.join(folder_path, file),
            cv.IMREAD_COLOR_RGB,
        )
        if img is None:
            print(f"Skipping unreadable file: {file}")
            continue

        result: np.ndarray = process_image(img, args)
        result = cv.cvtColor(result, cv.COLOR_RGB2BGR)

        name, _ = os.path.splitext(file)
        output_path: str = os.path.join(args.output, f"{name}.png")

        cv.imwrite(output_path, result, [cv.IMWRITE_PNG_COMPRESSION, 0])

        print(f"Finished {file} ({idx}/{total}) " f"in {time.time() - file_start:.2f}s")

    print(f"Finished processing {total} files " f"in {time.time() - batch_start:.2f}s")


if __name__ == "__main__":
    main()
