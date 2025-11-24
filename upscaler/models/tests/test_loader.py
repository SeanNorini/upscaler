import pytest
import torch

from upscaler.models.loader import ExtendedModelLoader


def test_build_model():
    """Test model construction without any file I/O."""

    class TestModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.state = None

        def load_state_dict(self, state, strict=True):
            self.state = state

    registry = {"arch": "TestModel", "param1": "value1"}
    state_dict = {"weight": torch.randn(10, 10)}

    loader = ExtendedModelLoader(
        model_specs={"test": registry},
        device=torch.device("cpu"),
        architecture_registry={"TestModel": TestModel},
    )

    model = loader._build_model(registry, state_dict)

    assert isinstance(model, TestModel)
    assert model.kwargs == registry
    assert model.state == state_dict


@pytest.mark.parametrize(
    "is_half,device_str", [(True, "cuda"), (False, "cpu"), (None, "cpu")]
)
def test_configure_model(is_half, device_str):
    """Test model configuration logic in isolation."""
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    class MockModel:
        def __init__(self):
            self.device = None
            self.is_eval = False
            self.is_half = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.is_eval = True
            return self

        def half(self):
            self.is_half = True
            return self

    loader = ExtendedModelLoader(model_specs={}, device=torch.device(device_str))

    model = MockModel()
    configured = loader._configure_model(model, {"is_half": is_half})

    assert configured.device == torch.device(device_str)
    assert configured.is_eval
    assert configured.is_half == (is_half if is_half else False)


def test_loader_uses_default_architectures():
    """Test that default architecture registry is used when none provided."""
    loader = ExtendedModelLoader(
        model_specs={},
        device=torch.device("cpu"),
    )

    assert "SwinUNet" in loader._architectures
    from upscaler.models.swin_unet import SwinUNet

    assert loader._architectures["SwinUNet"] is SwinUNet


def test_build_model_invalid_architecture():
    """Test error handling when architecture is not in registry."""
    registry = {"arch": "NonExistentModel"}
    state_dict = {}

    loader = ExtendedModelLoader(
        model_specs={"test": registry},
        device=torch.device("cpu"),
        architecture_registry={"SomeOtherModel": object},
    )

    with pytest.raises(KeyError):
        loader._build_model(registry, state_dict)
