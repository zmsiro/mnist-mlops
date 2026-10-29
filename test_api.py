"""Small tests for preprocessing, predictions, invalid files, and checkpoint loading."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest
import torch

from app import app, load_model
from model import LogisticRegression

client = TestClient(app)


def make_png():
    image = Image.new("L", (28, 28), 0)
    draw = ImageDraw.Draw(image)
    draw.line((14, 4, 14, 23), fill=255, width=3)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction_returns_valid_digit_and_probabilities():
    response = client.post("/predict/", files={"file": ("digit.png", make_png(), "image/png")})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert 0 <= payload["predicted_digit"] <= 9
    assert 0 <= payload["confidence"] <= 1
    assert len(payload["probabilities"]) == 10
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_invalid_image_rejected():
    response = client.post("/predict/", files={"file": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 400


def test_wrong_content_type_rejected():
    response = client.post("/predict/", files={"file": ("note.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_large_upload_rejected():
    response = client.post("/predict/", files={"file": ("big.png", b"A" * (5 * 1024 * 1024 + 1), "image/png")})
    assert response.status_code == 413


@pytest.mark.parametrize("hidden,dropout", [([128, 64], 0.2), ([256, 128], 0.3), ([], 0.0), ([64], 0.0)])
def test_checkpoint_architecture_detection(tmp_path, hidden, dropout):
    reference = LogisticRegression(784, 10, hidden, dropout)
    checkpoint = tmp_path / "model.pt"
    torch.save(reference.state_dict(), checkpoint)
    loaded = load_model(checkpoint)
    for name, expected in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected)
    assert not loaded.training
