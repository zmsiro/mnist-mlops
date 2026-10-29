
---

## Assignment 2 — FastAPI Deployment and Docker

This assignment extends the **same MNIST model and GitHub repository** from Assignment 1. The `app.py` service loads the trained `model.pt` weights, determines the fully connected architecture from the checkpoint, and returns a predicted digit (0–9), confidence, and probabilities. It uses the same grayscale 28 × 28 preprocessing and MNIST normalization used during training. Dark digits on a light background are inverted automatically.

### Run the API locally

From the repository root (Python 3.11 or 3.12 recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-api.txt
ls -lh model.pt
python -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs, expand `POST /predict/`, click **Try it out**, upload `sample_digit.png` (included) or another PNG/JPEG/WebP digit image, and click **Execute**. You will receive JSON with `predicted_digit`, `confidence`, and `probabilities`. The health endpoint is at http://127.0.0.1:8000/health.

To test via Terminal while the server is running:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@sample_digit.png"
```

Run automated tests:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

### Build and run with Docker

Install and start Docker Desktop first. From the **same repository root** (where `Dockerfile` and `model.pt` live):

```bash
docker build -t mnist-fastapi .
docker run --rm -p 8000:8000 mnist-fastapi
```

Open http://localhost:8000/docs and test `POST /predict/` with an image. Stop the container with Control+C. `model.pt` is copied into the image during `docker build`, and the API is available on port 8000. If port 8000 is taken, use `docker run --rm -p 8001:8000 mnist-fastapi` and open http://localhost:8001/docs.

### CI/CD (bonus)

`.github/workflows/deploy.yml` tests the API after pushes and pull requests to `main`. On pushes to `main`, it also builds and pushes a container to GitHub Container Registry (GHCR) using the automatically provided `GITHUB_TOKEN`. No personal access token needs to be committed. Publishing an image is **not** by itself a publicly hosted API. The optional Render deployment step runs only if a Render web service is configured separately and its deploy hook is saved as the `RENDER_DEPLOY_HOOK_URL` GitHub Actions repository secret.

### API files

- `sample_digit.png` — simple example image for trying the endpoint
- `app.py` — FastAPI service and MNIST image preprocessing
- `model.py` — same classifier architecture as Assignment 1
- `requirements-api.txt` — inference dependencies
- `requirements-test.txt` and `tests/test_api.py` — API testing
- `Dockerfile` and `.dockerignore` — container build
- `.github/workflows/deploy.yml` — automated test, Docker publishing, and optional deployment hook
- `model.pt` — existing trained weights from Assignment 1 (must be committed)

The Assignment 1 W&B report URL is retained in the **W&B report** section above.
