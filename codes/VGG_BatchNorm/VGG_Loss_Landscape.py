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
from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Light, get_number_of_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare VGG-A and VGG-A-BatchNorm on CIFAR-10."
    )
    parser.add_argument("--mode", type=str, default="comparison",
                        choices=["smoke", "comparison", "landscape", "ablation", "final"])
    parser.add_argument("--model", type=str, default="vgg_a_bn",
                        choices=["vgg_a", "vgg_a_bn", "vgg_a_light"],
                        help="Model used by final mode.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n_items", type=int, default=None,
                        help="Use a subset of CIFAR-10 for fast tests. Omit for full data.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lrs", type=float, nargs="+",
                        default=[1e-3, 2e-3, 5e-4, 1e-4],
                        help="Learning rates used by landscape mode.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--data_root", type=str, default=str(SCRIPT_DIR / "data"))
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional output tag for landscape runs, e.g. refined.")
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


def train(model, optimizer, criterion, train_loader, test_loader, device, epochs_n=1, show_progress=False):
    model.to(device)
    history = {
        "train_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs_n):
        model.train()
        running_loss = 0.0
        seen = 0

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{epochs_n}",
            unit="batch",
            leave=False,
            disable=not show_progress,
        )
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


def build_optimizer_by_name(parameters, optimizer_name, lr, weight_decay):
    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    return torch.optim.SGD(
        parameters,
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )


def build_optimizer(model, args, lr=None):
    lr = args.lr if lr is None else lr
    return build_optimizer_by_name(
        model.parameters(),
        optimizer_name=args.optimizer,
        lr=lr,
        weight_decay=args.weight_decay,
    )


def plot_training_curves(histories, output_path):
    epochs = range(1, len(next(iter(histories.values()))["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for name, history in histories.items():
        axes[0].plot(epochs, history["train_loss"], marker="o", label=name)
        axes[1].plot(epochs, history["test_accuracy"], marker="o", label=name)

    axes[0].set_title("Training Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Test Accuracy by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    for axis in axes:
        axis.legend()

    fig.suptitle("VGG-A vs VGG-A-BatchNorm on CIFAR-10", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_output_stem(base_name, tag):
    clean_tag = "".join(ch for ch in tag.strip() if ch.isalnum() or ch in ("_", "-"))
    return f"{base_name}_{clean_tag}" if clean_tag else base_name


def get_model_builders():
    return {
        "VGG_A": VGG_A,
        "VGG_A_BatchNorm": VGG_A_BatchNorm,
    }


def build_named_model(model_name):
    if model_name == "vgg_a":
        return "VGG_A", VGG_A()
    if model_name == "vgg_a_bn":
        return "VGG_A_BatchNorm", VGG_A_BatchNorm()
    if model_name == "vgg_a_light":
        return "VGG_A_Light", VGG_A_Light()
    raise ValueError(f"Unsupported model: {model_name}")


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
    model_builders = get_model_builders()
    output_name = "smoke" if args.mode == "smoke" else "comparison"

    for model_name, model_builder in model_builders.items():
        print(f"\nTraining {model_name}")
        set_random_seeds(args.seed, device)
        model = model_builder()
        optimizer = build_optimizer(model, args)
        criterion = nn.CrossEntropyLoss()
        history = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs_n=args.epochs,
            show_progress=args.show_progress,
        )
        histories[model_name] = history

        checkpoint_path = output_dirs["checkpoints"] / f"{model_name}_{output_name}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    metrics_path = output_dirs["results"] / f"vgg_batchnorm_{output_name}_metrics.json"
    save_json(
        {
            "args": vars(args),
            "epochs": list(range(1, args.epochs + 1)),
            "histories": histories,
        },
        metrics_path,
    )

    figure_path = output_dirs["pic"] / f"vgg_batchnorm_{output_name}_curves.png"
    plot_training_curves(histories, figure_path)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved figure: {figure_path}")
    return histories


def train_loss_curve(model, optimizer, criterion, train_loader, device, epochs_n=1, show_progress=False):
    model.to(device)
    epoch_losses = []

    for epoch in range(epochs_n):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{epochs_n}",
            unit="batch",
            leave=False,
            disable=not show_progress,
        )

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
        epoch_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs_n}: train_loss={train_loss:.4f}")

    return epoch_losses


def summarize_loss_variation(loss_by_lr):
    loss_matrix = np.array(list(loss_by_lr.values()), dtype=float)
    min_curve = loss_matrix.min(axis=0).tolist()
    max_curve = loss_matrix.max(axis=0).tolist()
    mean_curve = loss_matrix.mean(axis=0).tolist()
    variation_width = (loss_matrix.max(axis=0) - loss_matrix.min(axis=0)).tolist()
    return {
        "loss_by_lr": loss_by_lr,
        "min_curve": min_curve,
        "max_curve": max_curve,
        "mean_curve": mean_curve,
        "variation_width": variation_width,
        "average_variation_width": float(np.mean(variation_width)),
        "final_variation_width": float(variation_width[-1]) if variation_width else 0.0,
    }


def plot_loss_landscape(landscape_metrics, output_path):
    epochs = landscape_metrics["epochs"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {
        "VGG_A": "#d55e00",
        "VGG_A_BatchNorm": "#0072b2",
    }

    for model_name, metrics in landscape_metrics["models"].items():
        min_curve = metrics["min_curve"]
        max_curve = metrics["max_curve"]
        mean_curve = metrics["mean_curve"]
        color = colors.get(model_name)
        label = "VGG-A without BN" if model_name == "VGG_A" else "VGG-A with BatchNorm"
        ax.fill_between(
            epochs,
            min_curve,
            max_curve,
            color=color,
            alpha=0.22,
            label=f"{label} loss variation",
        )
        ax.plot(epochs, mean_curve, color=color, marker="o", label=f"{label} mean loss")

    ax.set_title("Loss Variation Across Learning Rates on CIFAR-10")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_loss_by_lr(landscape_metrics, output_path):
    epochs = landscape_metrics["epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    subplot_order = ["VGG_A", "VGG_A_BatchNorm"]
    subplot_titles = {
        "VGG_A": "VGG-A without BatchNorm",
        "VGG_A_BatchNorm": "VGG-A with BatchNorm",
    }

    for axis, model_name in zip(axes, subplot_order):
        metrics = landscape_metrics["models"][model_name]
        for lr, losses in metrics["loss_by_lr"].items():
            axis.plot(epochs, losses, marker="o", label=f"lr={lr}")
        axis.set_title(subplot_titles[model_name])
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[0].set_ylabel("Training Loss")
    fig.suptitle("Training Loss Curves for Each Learning Rate", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_landscape_summary(landscape_metrics):
    summary = {
        "args": landscape_metrics["args"],
        "epochs": landscape_metrics["epochs"],
        "models": {},
        "diagnosis": {},
    }

    for model_name, metrics in landscape_metrics["models"].items():
        final_loss_by_lr = {
            lr: losses[-1] for lr, losses in metrics["loss_by_lr"].items()
        }
        max_lr_by_epoch = []
        min_lr_by_epoch = []

        for epoch_index in range(len(landscape_metrics["epochs"])):
            losses_at_epoch = {
                lr: losses[epoch_index]
                for lr, losses in metrics["loss_by_lr"].items()
            }
            max_lr_by_epoch.append(max(losses_at_epoch, key=losses_at_epoch.get))
            min_lr_by_epoch.append(min(losses_at_epoch, key=losses_at_epoch.get))

        summary["models"][model_name] = {
            "final_loss_by_lr": final_loss_by_lr,
            "mean_loss_curve": metrics["mean_curve"],
            "variation_width": metrics["variation_width"],
            "average_variation_width": metrics["average_variation_width"],
            "final_variation_width": metrics["final_variation_width"],
            "max_loss_lr_by_epoch": max_lr_by_epoch,
            "min_loss_lr_by_epoch": min_lr_by_epoch,
        }

    bn_metrics = summary["models"].get("VGG_A_BatchNorm")
    if bn_metrics is not None:
        max_sources = sorted(set(bn_metrics["max_loss_lr_by_epoch"]))
        min_sources = sorted(set(bn_metrics["min_loss_lr_by_epoch"]))
        summary["diagnosis"]["bn_variation_sources"] = {
            "high_loss_learning_rates": max_sources,
            "low_loss_learning_rates": min_sources,
            "explanation": (
                "BN variation is widened by the gap between learning rates "
                "that keep loss high and learning rates that reduce loss much faster."
            ),
        }

    return summary


def run_loss_landscape(args):
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

    landscape_metrics = {
        "args": vars(args),
        "epochs": list(range(1, args.epochs + 1)),
        "lrs": args.lrs,
        "models": {},
    }

    criterion = nn.CrossEntropyLoss()
    for model_name, model_builder in get_model_builders().items():
        print(f"\nRunning loss landscape for {model_name}")
        loss_by_lr = {}

        for lr in args.lrs:
            print(f"Learning rate: {lr:g}")
            set_random_seeds(args.seed, device)
            model = model_builder()
            optimizer = build_optimizer(model, args, lr=lr)
            losses = train_loss_curve(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                device=device,
                epochs_n=args.epochs,
                show_progress=args.show_progress,
            )
            loss_by_lr[f"{lr:g}"] = losses

        landscape_metrics["models"][model_name] = summarize_loss_variation(loss_by_lr)

    output_stem = make_output_stem("vgg_batchnorm_loss_landscape", args.tag)
    metrics_path = output_dirs["results"] / f"{output_stem}_metrics.json"
    figure_path = output_dirs["pic"] / f"{output_stem}.png"
    by_lr_figure_path = output_dirs["pic"] / f"{output_stem}_by_lr.png"
    summary_path = output_dirs["results"] / f"{output_stem}_summary.json"
    save_json(landscape_metrics, metrics_path)
    plot_loss_landscape(landscape_metrics, figure_path)
    plot_loss_by_lr(landscape_metrics, by_lr_figure_path)
    save_json(build_landscape_summary(landscape_metrics), summary_path)

    print(f"Saved landscape metrics: {metrics_path}")
    print(f"Saved landscape figure: {figure_path}")
    print(f"Saved landscape by-lr figure: {by_lr_figure_path}")
    print(f"Saved landscape summary: {summary_path}")
    return landscape_metrics


def build_ablation_model(model_name, activation):
    if model_name == "VGG_A_Light":
        return VGG_A_Light(activation=activation)
    if model_name == "VGG_A":
        return VGG_A(activation=activation)
    raise ValueError(f"Unsupported ablation model: {model_name}")


def get_ablation_experiments(base_lr):
    return [
        {
            "experiment_name": "capacity_light_relu_adam",
            "model_name": "VGG_A_Light",
            "activation": "relu",
            "optimizer": "adam",
            "lr": base_lr,
            "weight_decay": 0.0,
        },
        {
            "experiment_name": "capacity_vgg_a_relu_adam",
            "model_name": "VGG_A",
            "activation": "relu",
            "optimizer": "adam",
            "lr": base_lr,
            "weight_decay": 0.0,
        },
        {
            "experiment_name": "activation_vgg_a_leakyrelu_adam",
            "model_name": "VGG_A",
            "activation": "leakyrelu",
            "optimizer": "adam",
            "lr": base_lr,
            "weight_decay": 0.0,
        },
        {
            "experiment_name": "optimizer_vgg_a_sgd_momentum",
            "model_name": "VGG_A",
            "activation": "relu",
            "optimizer": "sgd",
            "lr": 1e-2,
            "weight_decay": 0.0,
        },
        {
            "experiment_name": "regularization_vgg_a_adam_wd5e-4",
            "model_name": "VGG_A",
            "activation": "relu",
            "optimizer": "adam",
            "lr": base_lr,
            "weight_decay": 5e-4,
        },
    ]


def plot_ablation_summary(results, output_path):
    names = [result["experiment_name"] for result in results]
    accuracies = [result["best_test_accuracy"] for result in results]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars = ax.bar(range(len(results)), accuracies, color="#4c78a8")
    ax.set_title("CIFAR-10 Ablation: Best Test Accuracy")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Best Test Accuracy")
    ax.set_ylim(0.0, max(accuracies + [0.2]) + 0.08)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ablation_curves(histories, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for experiment_name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], marker="o", label=experiment_name)
        axes[1].plot(epochs, history["test_accuracy"], marker="o", label=experiment_name)

    axes[0].set_title("Ablation Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Ablation Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    for axis in axes:
        axis.legend(fontsize=8)

    fig.suptitle("CIFAR-10 Ablation Training Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_ablation(args):
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

    results = []
    histories = {}
    criterion = nn.CrossEntropyLoss()

    for experiment in get_ablation_experiments(args.lr):
        print(f"\nRunning ablation: {experiment['experiment_name']}")
        set_random_seeds(args.seed, device)
        model = build_ablation_model(
            model_name=experiment["model_name"],
            activation=experiment["activation"],
        )
        parameter_count = get_number_of_parameters(model)
        optimizer = build_optimizer_by_name(
            model.parameters(),
            optimizer_name=experiment["optimizer"],
            lr=experiment["lr"],
            weight_decay=experiment["weight_decay"],
        )
        history = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs_n=args.epochs,
            show_progress=args.show_progress,
        )
        histories[experiment["experiment_name"]] = history

        best_test_accuracy = max(history["test_accuracy"])
        result = {
            "experiment_name": experiment["experiment_name"],
            "model_name": experiment["model_name"],
            "activation": experiment["activation"],
            "optimizer": experiment["optimizer"],
            "lr": experiment["lr"],
            "weight_decay": experiment["weight_decay"],
            "n_items": args.n_items,
            "epochs": args.epochs,
            "parameter_count": parameter_count,
            "final_train_loss": history["train_loss"][-1],
            "final_test_accuracy": history["test_accuracy"][-1],
            "best_test_accuracy": best_test_accuracy,
            "best_test_error": 1.0 - best_test_accuracy,
        }
        results.append(result)
        print(
            f"{experiment['experiment_name']}: "
            f"best_test_accuracy={best_test_accuracy:.4f}, "
            f"best_test_error={1.0 - best_test_accuracy:.4f}"
        )

    metrics = {
        "args": vars(args),
        "results": results,
        "histories": histories,
    }
    metrics_path = output_dirs["results"] / "cifar10_ablation_results.json"
    summary_path = output_dirs["pic"] / "cifar10_ablation_summary.png"
    curves_path = output_dirs["pic"] / "cifar10_ablation_curves.png"

    save_json(metrics, metrics_path)
    plot_ablation_summary(results, summary_path)
    plot_ablation_curves(histories, curves_path)

    print(f"Saved ablation results: {metrics_path}")
    print(f"Saved ablation summary figure: {summary_path}")
    print(f"Saved ablation curves figure: {curves_path}")
    return metrics


def plot_final_training_curves(model_label, history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", color="#4c78a8")
    axes[0].set_title(f"{model_label} Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["test_accuracy"], marker="o", color="#59a14f")
    axes[1].set_title(f"{model_label} Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Final CIFAR-10 Training Run", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def final_checkpoint_name(model_name, tag=""):
    clean_tag = "".join(ch for ch in tag.strip() if ch.isalnum() or ch in ("_", "-"))
    suffix = f"_{clean_tag}" if clean_tag else ""
    if model_name == "vgg_a_bn":
        return f"final_vgg_a_bn{suffix}.pth"
    if model_name == "vgg_a_light":
        return f"final_vgg_a_light{suffix}.pth"
    return f"final_vgg_a{suffix}.pth"


def run_final_training(args):
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

    model_label, model = build_named_model(args.model)
    parameter_count = get_number_of_parameters(model)
    optimizer = build_optimizer(model, args)
    criterion = nn.CrossEntropyLoss()

    print(f"\nRunning final training: {model_label}")
    history = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs_n=args.epochs,
        show_progress=args.show_progress,
    )

    best_test_accuracy = max(history["test_accuracy"])
    best_epoch = history["test_accuracy"].index(best_test_accuracy) + 1
    result = {
        "model_name": model_label,
        "model_arg": args.model,
        "epochs": args.epochs,
        "n_items": args.n_items,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "final_train_loss": history["train_loss"][-1],
        "final_test_accuracy": history["test_accuracy"][-1],
        "best_test_accuracy": best_test_accuracy,
        "best_test_error": 1.0 - best_test_accuracy,
        "best_epoch": best_epoch,
    }

    metrics = {
        "args": vars(args),
        "result": result,
        "history": history,
    }
    output_stem = make_output_stem("final_training", args.tag)
    metrics_path = output_dirs["results"] / f"{output_stem}_results.json"
    figure_path = output_dirs["pic"] / f"{output_stem}_curves.png"
    checkpoint_path = output_dirs["checkpoints"] / final_checkpoint_name(args.model, args.tag)

    save_json(metrics, metrics_path)
    plot_final_training_curves(model_label, history, figure_path)
    torch.save(model.state_dict(), checkpoint_path)

    print(f"Saved final metrics: {metrics_path}")
    print(f"Saved final figure: {figure_path}")
    print(f"Saved final checkpoint: {checkpoint_path}")
    return metrics


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.mode == "landscape":
        run_loss_landscape(parsed_args)
    elif parsed_args.mode == "ablation":
        run_ablation(parsed_args)
    elif parsed_args.mode == "final":
        run_final_training(parsed_args)
    else:
        run_experiment(parsed_args)
