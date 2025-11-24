import numpy as np
import cv2 as cv
import pytest

from upscaler.utils import resize


@pytest.mark.parametrize(
    "shape,target_size,expected_max",
    [
        ((100, 100, 3), 200, 200),  # square
        ((50, 200, 3), 100, 100),  # wide image
        ((300, 100, 3), 150, 150),  # tall image
        ((2048, 2048, 3), 2048, 2048),  # already target size
    ],
)
def test_resize_max_side(shape, target_size, expected_max):
    img = np.zeros(shape, dtype=np.uint8)
    out = resize(img, target_size)

    h, w = out.shape[:2]
    assert max(h, w) == expected_max


def test_resize_aspect_ratio_preserved():
    img = np.zeros((120, 80, 3), dtype=np.uint8)
    out = resize(img, 240)

    h, w = out.shape[:2]
    assert h / w == pytest.approx(120 / 80)


def test_resize_preserves_dtype():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = resize(img, 500)

    assert out.dtype == np.uint8
