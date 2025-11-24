import json
from pathlib import Path
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
import torch.nn.functional as F

from upscaler.models.loader import ExtendedModelLoader
from upscaler.models.wrapper import ModelWrapper
from upscaler.utils import resize, to_tensor, to_rgb, SeamBlending

_upscaler = None


class Upscaler:
    def __init__(self, model_specs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_specs = model_specs
        self._model_loader = ExtendedModelLoader(model_specs, self.device)
        self._models: dict[str, ModelWrapper] = {}

    def _get_model_data(self, model_variant: str) -> dict[str, any]:
        if model_variant not in self._model_specs:
            raise ValueError(f"Unsupported model request: {model_variant}")
        return self._model_specs[model_variant]

    def _load_model(self, model_name: str) -> ModelWrapper:
        if model_name in self._models:
            return self._models[model_name]

        model = self._model_loader.load_from_file(model_name=model_name)
        self._models[model_name] = model
        return model

    def _render(
        self,
        x,
        model_variant: str,
        enable_amp: bool,
    ) -> Tensor:
        model = self._load_model(model_variant)
        x = F.pad(x, (model.pre_pad,) * 4, "constant", 1)

        rgb = SeamBlending.tiled_render(x, model)

        crop = model.pre_pad * model.scale
        rgb = rgb[:, crop:-crop, crop:-crop]

        return rgb

    @torch.inference_mode()
    def process(
        self,
        img: NDArray[np.uint8],
        model_variant: str,
        output_device: str = "cpu",
    ) -> NDArray[np.uint8]:
        rgb_tensor = to_tensor(img)

        assert not torch.is_grad_enabled()
        assert rgb_tensor.shape[0] == 3

        rgb_tensor = rgb_tensor.to(self.device)
        rgb = self._render(rgb_tensor, model_variant).to(output_device)

        return to_rgb(rgb)

    def __call__(self, img, model_name):
        return self.process(img, model_name)


def upscale(
    img: NDArray[np.uint8], model_name: str, target_size: int = 0
) -> NDArray[np.uint8]:
    upscaler = _get_upscaler()
    curr_size = -1
    while curr_size < target_size:
        img = upscaler(img, model_name)
        curr_size = max(img.shape[:2])

    if target_size != 0 and curr_size > target_size:
        img = resize(img, target_size)

    return img.astype(np.uint8)


def _get_upscaler():
    global _upscaler
    if _upscaler is None:
        model_specs = _get_model_specs()
        _upscaler = Upscaler(model_specs)
    return _upscaler


def _get_model_specs():
    directory = Path(__file__).resolve().parent
    with open(directory / "model_specs.json") as f:
        model_specs = json.load(f)
        return model_specs
