"""FastAPI service for the exact MNIST weights saved by Assignment 1.

Run locally: uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from io import BytesIO
from pathlib import Path
import re

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError
import torch

from model import LogisticRegression

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}


def load_model(path: Path = MODEL_PATH) -> LogisticRegression:
    """Determine the saved layer widths (and dropout layout) from state_dict keys.

    The original train.py saves only state_dict, not the hyperparameters.
    This handles baseline, wide-MLP, or no-hidden-layer checkpoints.
    Dropout is inactive during inference, so any nonzero rate can reproduce
    the original layout and state_dict keys.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at {path}. Copy Assignment 1 model.pt "
            "into the project root before starting the API."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected a PyTorch state_dict in model.pt")

    linear_layers = []
    for key, value in state.items():
        match = re.fullmatch(r"network\.(\d+)\.weight", key)
        if match and isinstance(value, torch.Tensor) and value.ndim == 2:
            linear_layers.append((int(match.group(1)), value))
    linear_layers.sort(key=lambda pair: pair[0])

    if not linear_layers:
        raise ValueError("model.pt has no matching network.*.weight layers")
    indices = [i for i, _ in linear_layers]
    dropout_rate = 0.2 if len(indices) > 1 and indices[1] == 3 else 0.0
    expected_indices = list(range(0, 3 * len(indices), 3)) if dropout_rate else list(range(0, 2 * len(indices), 2))
    if indices != expected_indices:
        raise ValueError(f"Unexpected linear-layer indices {indices}; inspect training architecture")
    input_dim = linear_layers[0][1].shape[1]
    output_dim = linear_layers[-1][1].shape[0]
    if input_dim != 784 or output_dim != 10:
        raise ValueError(f"Expected MNIST model with 784 inputs and 10 outputs; found {input_dim}/{output_dim}")
    hidden_layers = [weight.shape[0] for _, weight in linear_layers[:-1]]
    model = LogisticRegression(784, 10, hidden_layers, dropout_rate)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# Fail early and clearly if the actual trained weights are missing.
model = load_model()
app = FastAPI(title="MNIST Digit Classifier", version="2.0.0")


@app.get("/")
def home():
    return {"message": "MNIST prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WebP image")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be no larger than 5 MB")
    if not contents:
        raise HTTPException(status_code=400, detail="The image file is empty")

    try:
        with Image.open(BytesIO(contents)) as uploaded:
            if uploaded.width * uploaded.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image has too many pixels")
            uploaded.load()
            image = uploaded.convert("L")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Could not decode a valid image") from exc

    # MNIST images use bright strokes on a black background. Invert light-
    # background photographs/screenshots so their convention is similar.
    if ImageStat.Stat(image).mean[0] > 127:
        image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.BILINEAR)

    pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.1307) / 0.3081  # Same MNIST normalization as train.py

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    digit = int(probabilities.argmax().item())
    return {
        "predicted_digit": digit,
        "confidence": round(float(probabilities[digit]), 6),
        "probabilities": {str(i): round(float(p), 6) for i, p in enumerate(probabilities)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
