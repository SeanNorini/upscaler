import numpy as np
import torch
import pytest

from image_enhancement.upscaler.utils import to_tensor, to_rgb


@pytest.mark.parametrize("shape", [(32, 32, 3), (10, 40, 3), (128, 64, 3)])
def test_to_tensor_shape_and_dtype(shape):
    img = np.zeros(shape, dtype=np.uint8)
    t = to_tensor(img)

    assert t.shape == (3, shape[0], shape[1])
    assert t.dtype == torch.float32


@pytest.mark.parametrize("shape", [(32, 32, 3), (10, 40, 3)])
def test_to_rgb_shape_and_dtype(shape):
    tensor = torch.zeros((3, shape[0], shape[1]), dtype=torch.float32)
    img = to_rgb(tensor)

    assert img.shape == shape
    assert img.dtype == np.uint8


def test_round_trip():
    img = np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8)

    t = to_tensor(img)
    recovered = to_rgb(t)

    assert recovered.shape == img.shape
    # round trip must be close but allow off-by-1 from float rounding
    assert np.abs(recovered.astype(np.int16) - img.astype(np.int16)).max() <= 1
