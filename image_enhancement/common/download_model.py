import requests
from pathlib import Path


def download_model(model_name: str, model_path: Path, spec: dict):
    url = spec.get("download_url")
    if url is None:
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path} "
            f"and no 'download_url' provided."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from {url} to {model_path}...")

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    model_path.write_bytes(r.content)
