import numpy as np
from numpy.typing import NDArray
import cv2 as cv


def resize(img: NDArray[np.uint8], target_size: int = 2048) -> NDArray[np.uint8]:
    """Resize an image to scale the largest side to the target size while maintaining the aspect ratio."""
    orig_h, orig_w = img.shape[:2]
    factor = target_size / max(orig_h, orig_w)
    return cv.resize(img, None, fx=factor, fy=factor, interpolation=cv.INTER_CUBIC)
