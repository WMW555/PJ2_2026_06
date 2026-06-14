import argparse
import math
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.vgg import VGG_A, VGG_A_BatchNorm


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize first-layer Conv2d filters.")
    parser.add_argument("--model", type=str, default="vgg_a_bn",
                        choices=["vgg_a", "vgg_a_bn"])
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "checkpoints" / "final_vgg_a_bn_regularized.pth"))
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "pic" / "first_layer_filters_regularized.png"))
    parser.add_argument("--max_filters", type=int, default=64)
    return parser.parse_args()


def build_model(model_name):
    if model_name == "vgg_a":
        return VGG_A()
    if model_name == "vgg_a_bn":
        return VGG_A_BatchNorm()
    raise ValueError(f"Unsupported model: {model_name}")


def find_first_conv2d(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model.")


def normalize_filter(filter_tensor):
    image = filter_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    min_value = image.min()
    max_value = image.max()
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image)
    return image


def plot_filters(filters, output_path, max_filters):
    filters = filters[:max_filters]
    count = filters.shape[0]
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.25))
    axes = np.array(axes).reshape(rows, cols)

    for index in range(rows * cols):
        axis = axes[index // cols, index % cols]
        axis.axis("off")
        if index < count:
            axis.imshow(normalize_filter(filters[index]))
            axis.set_title(str(index), fontsize=7)

    fig.suptitle("First-Layer Convolution Filters", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return count


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run final regularized training first, or pass --checkpoint explicitly."
        )

    model = build_model(args.model)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    first_conv = find_first_conv2d(model)
    filter_count = plot_filters(first_conv.weight.data, output_path, args.max_filters)
    print(f"Saved {filter_count} first-layer filters to {output_path}")


if __name__ == "__main__":
    main()
