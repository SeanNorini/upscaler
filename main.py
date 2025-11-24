import argparse
import os
import time
import cv2 as cv

from upscaler import upscale


def get_args():
    parser = argparse.ArgumentParser(description="Image upscaler")
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
    return parser.parse_args()


def get_files(path):
    if os.path.isdir(path):
        folder_path = path
        files = os.listdir(path)
    else:
        folder_path = os.path.dirname(path)
        files = [os.path.basename(path)]
    return folder_path, files


def upscale_image(img, model_name, output_path):
    img = upscale(img, model_name, 2048)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(output_path, img, [cv.IMWRITE_PNG_COMPRESSION, 0])


def main():
    args = get_args()
    folder_path, files = get_files(args.input)

    count = 1
    total = len(files)
    batch_start = time.time()
    for file in files:
        file_start = time.time()
        img = cv.imread(os.path.join(folder_path, file), cv.IMREAD_COLOR_RGB)
        file_name, file_extension = os.path.splitext(file)
        output_path = os.path.join(args.output, file_name)

        upscale_image(img, args.model, output_path + ".png")

        print(
            f"Finished processing {file} ({count}/{total}). Processing took {time.time() - file_start} seconds."
        )
        count += 1

    print(
        f"Finished processing {total} files. Processing took {time.time() - batch_start} seconds."
    )


if __name__ == "__main__":
    main()
