import numpy as np
import torch


def image_to_tensor(img):
    """
    Convert RGB uint8 image to tensor.
    """
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to RGB uint8 image.
    """
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # HWC
    return np.uint8((img * 255.0).round())
