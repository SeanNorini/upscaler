import os
from pathlib import Path
import torch
from spandrel import ModelLoader

from upscaler.models.swin_unet import SwinUNet

_architectures = {"SwinUNet": SwinUNet}


class ExtendedModelLoader(ModelLoader):
    def __init__(
        self,
        model_specs: dict[str, dict[str, any]],
        device: torch.device,
        architecture_registry: dict[str, type] | None = None,
        weights_only: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._registry = model_specs
        self._map_location = device
        self._weights_only = weights_only
        self._architectures = architecture_registry or {"SwinUNet": SwinUNet}

        directory = Path(__file__).resolve().parent
        self._folder_path = Path.joinpath(directory, "weights")

    def load_from_file(self, model_name, **kwargs):
        data = self._registry[model_name]
        model_path = os.path.join(self._folder_path, model_name + data["ext"])
        if data["arch"] == "Spandrel":
            model = super().load_from_file(model_path).model
        else:
            state_dict = torch.load(
                model_path,
                map_location=self._map_location,
                weights_only=self._weights_only,
            )
            model = self._build_model(data, state_dict)

        return self._configure_model(model, data)

    def _build_model(self, data: dict, state_dict: dict):
        """Build model from architecture and state dict."""
        model_arch = self._architectures[data["arch"]]
        model = model_arch(**data)
        model.load_state_dict(state_dict, strict=True)
        return model

    def _configure_model(self, model, data: dict):
        """Apply device, eval, and half precision settings."""
        model = model.to(self._map_location).eval()
        if data.get("is_half", False):
            model = model.half()
        return model
