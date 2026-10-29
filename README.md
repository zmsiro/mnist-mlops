# MNIST Classifier — PyTorch + Weights & Biases

## Project overview

This assignment trains a classifier for 28 × 28 grayscale handwritten MNIST digits (0–9) using PyTorch. The `LogisticRegression` class preserves the name in the assignment skeleton. With `hidden_layers=[]`, it is logistic regression; with hidden layers (the default), it is a multilayer perceptron (MLP) with ReLU activations and dropout.

The pipeline records experiments in [Weights & Biases](https://wandb.ai/):

- Per-batch training loss, learning rate, and L2 gradient norms
- Per-epoch training and validation losses
- Per-epoch validation accuracy, macro precision, macro recall, and macro F1
- Final confusion matrix (interactive chart and report-ready image)
- A table of example predictions (images, true labels, predicted labels, confidence, correctness)
- A 10-image prediction grid displayed in an automatically generated W&B report
- The trained weights as both `model.pt` and a W&B model artifact

## Installation and local execution

Requirements: Python 3.10+ and an internet connection for the initial MNIST download and W&B synchronization.

```bash
# Clone your GitHub repository, then enter the project directory.
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
wandb login
python train.py
```

The first run downloads MNIST to `./data/`. After 10 epochs, the script writes `model.pt` and `prediction_grid.png` and `confusion_matrix.png` in this directory, sends training metrics, images and a model artifact to W&B, and attempts to publish a report automatically. If the report is created, `report_url.txt` and this README are updated with its URL.

## Example hyperparameter experiments

Run **at least two or three configurations** to compare architectures and settings in the W&B project `mnist-mlops`:

```bash
# Default: 2 hidden layers, LR 0.01, dropout 0.2
python train.py

# Experiment 2: wider network with lower LR
python train.py --learning-rate 0.001 --hidden-layers 256,128 --dropout-rate 0.3

# Experiment 3: no hidden layers (true logistic regression)
python train.py --learning-rate 0.001 --hidden-layers none --dropout-rate 0
```

Other options: `--epochs 5`, `--batch-size 128`, `--seed 123`, `--project my-project`, `--log-every 20`, and `--no-report`. Each run is visible separately in W&B. Note that `model.pt` is overwritten by each training run; retain the weights from your selected run for the submission (W&B also keeps a versioned artifact per run).

## W&B report

https://forge.coreweave.com/wandb/zmsiro-iowa-state-university/mnist-mlops/reports/MNIST-Model-Training---k8fttlsw--VmlldzoxODIwNDIzNg==

ChatGPT (OpenAI) was used to assist with code development, debugging, and documentation. AI-generated suggestions were reviewed and incorporated into the project, with model training and experiment tracking performed using PyTorch and Weights & Biases.
