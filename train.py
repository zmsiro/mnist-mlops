"""MNIST classifier with PyTorch, experiment tracking, and a W&B report.

Usage:
    wandb login
    python train.py
    python train.py --learning-rate 0.001 --hidden-layers 256,128 --dropout-rate 0.3
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Work on servers without a screen.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support,
)
from torchvision import datasets, transforms
import wandb


class LogisticRegression(nn.Module):
    """Configurable fully connected classifier (an MLP with hidden layers).

    The class name matches the assignment skeleton. With no hidden layers it
    is a true multinomial logistic regression; otherwise it is an MLP.
    """

    def __init__(self, input_dim, output_dim, hidden_layers=(128, 64), dropout_rate=0.2):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # [batch, 1, 28, 28] -> [batch, 784]
        return self.network(x)  # Raw logits, not probabilities.


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MNIST model with W&B")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-layers", default="128,64",
                        help="Comma-separated layer widths, or 'none' for logistic regression")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log batch-level metrics every N batches")
    parser.add_argument("--project", default="mnist-mlops")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip automatic W&B report creation")
    return parser.parse_args()


def parse_hidden_layers(value):
    if value.strip().lower() in ("", "none", "0"):
        return []
    sizes = [int(piece.strip()) for piece in value.split(",")]
    if not sizes or any(width <= 0 for width in sizes):
        raise ValueError("--hidden-layers must contain positive integers or 'none'")
    return sizes


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, run, epoch,
                    global_step, log_every):
    model.train()
    total_loss, total_seen = 0.0, 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        # Returns the original L2 norm and clips any unusually large gradients.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size
        global_step += 1

        if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
            run.log({
                "global_step": global_step,
                "train/batch_loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/gradient_norm": float(grad_norm),
            })

    return total_loss / total_seen, global_step


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            predictions = torch.argmax(logits, dim=1)
            total_loss += loss.item() * target.size(0)
            total_correct += (predictions == target).sum().item()
            total_seen += target.size(0)
            all_targets.extend(target.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, labels=list(range(10)),
        average="macro", zero_division=0,
    )
    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "targets": all_targets,
        "predictions": all_predictions,
    }


def log_predictions(model, data, target, device, run, num_samples=10):
    """Log prediction examples as a rich W&B Table and a report-ready PNG."""
    model.eval()
    data, target = data[:num_samples], target[:num_samples]
    with torch.no_grad():
        probabilities = torch.softmax(model(data.to(device)), dim=1).cpu()
    confidence, predicted = probabilities.max(dim=1)
    images = (data.cpu() * 0.3081 + 0.1307).clamp(0, 1)  # Undo normalization.

    table = wandb.Table(columns=["image", "true_digit", "predicted_digit", "confidence", "correct"])
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(min(num_samples, len(data))):
        truth, guess = int(target[i]), int(predicted[i])
        table.add_data(
            wandb.Image(images[i], caption=f"True: {truth}, Predicted: {guess}"),
            truth, guess, float(confidence[i]), bool(truth == guess),
        )
        ax = axes.flat[i]
        ax.imshow(images[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"True {truth} | Pred {guess}\n{confidence[i]:.1%} confident")
        ax.axis("off")
    for ax in axes.flat[min(num_samples, len(data)):]:
        ax.axis("off")
    fig.tight_layout()
    image_path = Path("prediction_grid.png")
    fig.savefig(image_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    run.log({"predictions/samples": table,
             "predictions/grid": wandb.Image(str(image_path), caption="MNIST predictions")})
    run.save(str(image_path), policy="now")
    return image_path


def log_confusion_matrix_image(results, run):
    """Save a static confusion matrix as well as the interactive W&B chart."""
    matrix = confusion_matrix(
        results["targets"], results["predictions"], labels=list(range(10)),
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=list(range(10)),
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("MNIST validation confusion matrix")
    fig.tight_layout()
    path = Path("confusion_matrix.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    run.log({"val/confusion_matrix_image": wandb.Image(str(path))})
    run.save(str(path), policy="now")
    return path


def create_report(run, image_path, confusion_path):
    """Publish a report with live charts and the saved prediction image."""
    import wandb_workspaces.reports.v2 as wr

    # This link is to the original PNG uploaded as a file of this run.
    image_url = f"{run.url}/files/{image_path.name}"
    confusion_url = f"{run.url}/files/{confusion_path.name}"
    report = wr.Report(
        entity=run.entity,
        project=run.project,
        title=f"MNIST Model Training - {run.id}",
        description="PyTorch MNIST classifier, training metrics, and example predictions",
        blocks=[
            wr.H1("MNIST model training and experiment tracking"),
            wr.P(f"Run: {run.url}"),
            wr.H2("Training and validation"),
            wr.PanelGrid(panels=[
                wr.LinePlot(x="epoch", y=["train/epoch_loss", "val/loss"],
                            title="Training and validation loss"),
                wr.LinePlot(x="epoch", y=["val/accuracy"], title="Validation accuracy"),
                wr.LinePlot(x="epoch", y=["val/precision", "val/recall", "val/f1"],
                            title="Macro precision, recall, F1"),
                wr.LinePlot(x="global_step", y=["train/gradient_norm"],
                            title="Gradient norm"),
                wr.LinePlot(x="global_step", y=["train/learning_rate"],
                            title="Learning rate"),
            ]),
            wr.H2("Sample predictions"),
            wr.P("Ten validation images with true labels, predictions, and confidence scores. "
                 f"An interactive predictions table and confusion matrix are logged in {run.url}."),
            wr.Image(url=image_url, caption="Predicted and true handwritten digits"),
            wr.H2("Confusion matrix"),
            wr.Image(url=confusion_url, caption="Final validation confusion matrix (0–9)"),
        ],
    )
    report.save()
    return getattr(report, "url", None)


def update_readme_report_url(url):
    if not url:
        return
    path = Path(__file__).resolve().parent / "README.md"
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    start, end = "<!-- REPORT_URL_START -->", "<!-- REPORT_URL_END -->"
    if start in content and end in content:
        before, remaining = content.split(start, 1)
        _, after = remaining.split(end, 1)
        content = before + start + f"\nW&B report: {url}\n" + end + after
        path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    if args.epochs <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("epochs, batch-size, and log-every must all be positive")
    if not 0 <= args.dropout_rate < 1 or args.learning_rate <= 0:
        raise ValueError("dropout-rate must be in [0, 1); learning-rate must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_layers": hidden_layers,
        "dropout_rate": args.dropout_rate,
        "seed": args.seed,
        "optimizer": "Adam",
    }
    run = wandb.init(project=args.project, config=config, job_type="train")
    config = wandb.config
    device = get_device()
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
    )

    model = LogisticRegression(
        input_dim=28 * 28, output_dim=10,
        hidden_layers=config.hidden_layers, dropout_rate=config.dropout_rate,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    wandb.watch(model, log="all", log_freq=100)
    wandb.define_metric("epoch")
    wandb.define_metric("global_step")
    wandb.define_metric("train/epoch_loss", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("train/batch_loss", step_metric="global_step")
    wandb.define_metric("train/gradient_norm", step_metric="global_step")
    wandb.define_metric("train/learning_rate", step_metric="global_step")

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            run, epoch, global_step, args.log_every,
        )
        results = evaluate(model, test_loader, criterion, device)
        run.log({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "val/loss": results["loss"],
            "val/accuracy": results["accuracy"],
            "val/precision": results["precision"],
            "val/recall": results["recall"],
            "val/f1": results["f1"],
        })
        print(f"Epoch {epoch:02d}/{config.epochs}: train loss={train_loss:.4f}, "
              f"val loss={results['loss']:.4f}, val accuracy={results['accuracy']:.2%}, "
              f"macro F1={results['f1']:.4f}")

    # Confusion matrix for the final trained model.
    run.log({"val/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, y_true=results["targets"], preds=results["predictions"],
        class_names=[str(i) for i in range(10)],
    )})

    confusion_path = log_confusion_matrix_image(results, run)

    # Use a single batch for images and labels to keep them aligned.
    sample_data, sample_labels = next(iter(test_loader))
    image_path = log_predictions(model, sample_data, sample_labels, device, run)

    # Exact file name specified by the assignment.
    torch.save(model.state_dict(), "model.pt")
    artifact = wandb.Artifact(f"mnist-model-{run.id}", type="model")
    artifact.add_file("model.pt")
    run.log_artifact(artifact)
    run.summary["final_validation_accuracy"] = results["accuracy"]
    run.summary["final_validation_f1"] = results["f1"]
    print("Saved model.pt")

    # Finish syncing uploads before creating a report that references them.
    run.finish()
    if not args.no_report:
        try:
            report_url = create_report(run, image_path, confusion_path)
            if report_url:
                print(f"W&B report: {report_url}")
                Path("report_url.txt").write_text(report_url + "\n", encoding="utf-8")
                update_readme_report_url(report_url)
            else:
                print("Report saved, but the API did not return a URL. Copy it from W&B Reports.")
        except Exception as exc:
            print(f"Automatic report creation failed: {exc}")
            print("Your training run, logged media, and model.pt should still be available.")


if __name__ == "__main__":
    main()
