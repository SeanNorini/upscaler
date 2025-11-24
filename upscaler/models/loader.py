import os
from pathlib import Path
from typing import Optional

import torch
from spandrel import ModelLoader

from upscaler.models.swin_unet import SwinUNet
from upscaler.models.wrapper import ModelWrapper

_architectures = {"SwinUNet": SwinUNet}


class ExtendedModelLoader(ModelLoader):
    def __init__(
        self,
        model_specs: dict[str, dict[str, any]],
        device: torch.device,
        architecture_registry: Optional[dict[str, type]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._registry = model_specs
        self._map_location = device
        self._architectures = architecture_registry or {"SwinUNet": SwinUNet}

        directory = Path(__file__).resolve().parent
        self._folder_path = Path.joinpath(directory, "weights")

    def load_from_file(self, model_name, **kwargs) -> ModelWrapper:
        spec = self._registry[model_name]
        load_cfg = spec["load"]
        config_cfg = spec.get("config", {})

        model_path = os.path.join(
            self._folder_path, model_name + load_cfg.get("ext", ".pth")
        )
        if load_cfg["arch"] == "Spandrel":
            model = super().load_from_file(model_path).model
        else:
            state_dict = torch.load(
                model_path, map_location=self._map_location, weights_only=True
            )
            model = self._build_model(load_cfg, state_dict)

        model = self._configure_model(model, load_cfg)
        return ModelWrapper(model, **config_cfg)

    def _build_model(self, cfg: dict, state_dict: dict):
        """Build model from architecture and state dict."""
        model_arch = self._architectures[cfg["arch"]]
        model = model_arch(**cfg)
        model.load_state_dict(state_dict, strict=True)
        return model

    def _configure_model(self, model, cfg: dict):
        """Apply device, eval, and half precision settings."""
        model = model.to(self._map_location).eval()
        if cfg.get("is_half", True):
            model = model.half()
        return model
