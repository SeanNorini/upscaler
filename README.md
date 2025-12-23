# Upscaler

A lightweight image upscaling toolkit built around PyTorch, Real-ESRGAN-style models, and a tiled-rendering system designed for large images.  
This project provides:

- A flexible **Upscaler** class for loading and running multiple models
- Automatic **tiling + seam blending** for memory-efficient inference
- A simple **CLI tool** for batch upscaling images
- Support for custom models defined in `model_specs.json`
- Optional **auto-download** support for missing model weights

---

## Features

### Multi-model support
Define models and their configs in `model_specs.json`, then upscale using:

```python
from image_enhancement.upscaler import upscale

result = upscale(img, "Real_ESRGAN_Video_4x")
```

### Command-line interface
Upscale single images or folders:

```bash
python main.py -i ./images -o ./output -m Real_ESRGAN_Video_4x -s 2048
```

### Automatic resizing
If a `target_size` is provided, the longest image side is cropped down after the final upscale.

---

## Usage

### **Python API**

```python
from image_enhancement.upscaler import upscale
import cv2 as cv

img = cv.imread("input.png", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

out = upscale(img, "Real_ESRGAN_Video_4x", target_size=2048)
```

### **CLI**

```bash
python main.py \
  --input ./images \
  --output ./output \
  --model Real_ESRGAN_Video_4x \
  --size 2048
```

---

## Project Structure

```
upscaler/
    models/
        loader.py      # extended ModelLoader with download support
        wrapper.py     # wraps model properties (scale, pad, etc.)
    utils/
        resize.py
        to_tensor.py
        SeamBlending.py
main.py                # CLI entrypoint
model_specs.json       # model config + metadata
```

---

## Model Weights & Licensing

This project can load third-party models (ESRGAN, Waifu, CUGAN, etc.).  
Each model retains its original license.  
Place licenses in:

```
licenses/
    Real_ESRGAN_LICENSE.txt
    Waifu2x_LICENSE.txt
    ...
```

You may host weights yourself or use auto-download URLs defined in `model_specs.json`.

---

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- NumPy
- spandrel (for model loading)

Install dependencies using Poetry:

```bash
poetry install
```

---

## Running with Poetry

```bash
poetry run python main.py
```
