from pathlib import Path
import numpy as np
import torch

from .io import image_to_tensor, tensor_to_rgb
from .model.network_scunet import SCUNet
from ..common.download_model import download_model


def load_model(model_name, model_path, device, n_channels: int = 3):
    model = SCUNet(in_nc=n_channels, config=[4, 4, 4, 4, 4, 4, 4], dim=64)

    if not model_path.exists():
        spec = {
            "download_url": f"https://github.com/cszn/KAIR/releases/download/v1.0/{model_name}.pth"
        }
        download_model(model_name, model_path, spec)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


def get_model_path(strength: int) -> Path:
    return (
        Path(__file__).resolve().parent / f"model/weights/scunet_color_{strength}.pth"
    )


def denoise(img, strength: int = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = get_model_path(strength)
    model = load_model(f"scunet_color_{strength}", model_path, device)
    img = image_to_tensor(img).to(device)
    with torch.no_grad():
        out = model(img)
    return tensor_to_rgb(out)
