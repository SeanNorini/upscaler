import numpy as np
import torch


def image_to_tensor(img):
    img = np.float32(img / 255.0)
    return (
        torch.from_numpy(np.ascontiguousarray(img))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
    )


def tensor_to_rgb(img):
    img = img.detach().squeeze().float().clamp(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())
