from functools import lru_cache


class ModelWrapper:
    """
    A wrapper for PyTorch models that stores metadata required for tiling and processing.
    """

    def __init__(
        self,
        model,
        scale=4,
        is_half=True,
        offset=0,
        pre_pad=8,
        blend_size=4,
        tile_size=256,
        batch_size=4,
    ):
        self.model = model
        self.scale = scale
        self.is_half = is_half
        self.offset = offset * scale
        self.pre_pad = pre_pad * scale
        self.blend_size = blend_size * scale
        self.default_tile_size = tile_size
        self.default_batch_size = batch_size

    def __getattr__(self, name):
        model = self.__dict__.get("model", None)
        if model is not None:
            return getattr(model, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")

    def __call__(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def get_device(self):
        """
        Gets the device of the underlying model's parameters.
        """
        return next(self.model.parameters()).device

    @lru_cache
    def find_valid_tile_size(self):
        """
        Finds a valid tile size for the model based on its constraints.
        This method is here as an example and might need to be adjusted
        based on the specific requirements of the models you load.
        """
        tile_size = self.default_tile_size
        while tile_size > 0:
            if (
                tile_size > 16
                and (tile_size - 16) % 12 == 0
                and (tile_size - 16) % 16 == 0
            ):
                return tile_size
            tile_size -= 1
        raise ValueError(
            f"Could not find valid tile size: tile_size={self.default_tile_size}"
        )
