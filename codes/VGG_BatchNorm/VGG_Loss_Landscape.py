import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test VGG-A and VGG-A-BatchNorm on CIFAR-10."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n_items", type=int, default=None,
                        help="Use a subset of CIFAR-10 for fast tests. Omit for full data.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--data_root", type=str, default=str(SCRIPT_DIR / "data"))
    return parser.parse_args()


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    return device


def set_random_seeds(seed_value=0, device=None):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_output_dirs():
    paths = {
        "pic": PROJECT_ROOT / "pic",
        "checkpoints": PROJECT_ROOT / "checkpoints",
        "results": PROJECT_ROOT / "results",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            predicted_labels = prediction.argmax(dim=1)
            correct += (predicted_labels == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def train(model, optimizer, criterion, train_loader, test_loader, device, epochs_n=1):
    model.to(device)
    history = {
        "train_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs_n):
        model.train()
        running_loss = 0.0
        seen = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs_n}", unit="batch")
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            progress.set_postfix(loss=running_loss / max(seen, 1))

        train_loss = running_loss / max(seen, 1)
        test_accuracy = get_accuracy(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["test_accuracy"].append(test_accuracy)
        print(
            f"Epoch {epoch + 1}/{epochs_n}: "
            f"train_loss={train_loss:.4f}, test_accuracy={test_accuracy:.4f}"
        )

    return history


def plot_training_curves(histories, output_path):
    epochs = range(1, len(next(iter(histories.values()))["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for name, history in histories.items():
        axes[0].plot(epochs, history["train_loss"], marker="o", label=name)
        axes[1].plot(epochs, history["test_accuracy"], marker="o", label=name)

    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    for axis in axes:
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_experiment(args):
    output_dirs = ensure_output_dirs()
    device = get_device()
    set_random_seeds(args.seed, device)

    train_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=True,
        shuffle=True,
        num_workers=args.num_workers,
        n_items=args.n_items,
    )
    test_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.n_items,
    )

    histories = {}
    model_builders = {
        "VGG_A": VGG_A,
        "VGG_A_BatchNorm": VGG_A_BatchNorm,
    }

    for model_name, model_builder in model_builders.items():
        print(f"\nTraining {model_name}")
        set_random_seeds(args.seed, device)
        model = model_builder()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        history = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs_n=args.epochs,
        )
        histories[model_name] = history

        checkpoint_path = output_dirs["checkpoints"] / f"{model_name}_smoke.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    metrics_path = output_dirs["results"] / "vgg_batchnorm_smoke_metrics.json"
    save_json(
        {
            "args": vars(args),
            "histories": histories,
        },
        metrics_path,
    )

    figure_path = output_dirs["pic"] / "vgg_batchnorm_smoke_curves.png"
    plot_training_curves(histories, figure_path)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved figure: {figure_path}")
    return histories


def plot_loss_landscape():
    raise NotImplementedError("Loss landscape is intentionally left for the next task.")


if __name__ == "__main__":
    run_experiment(parse_args())
