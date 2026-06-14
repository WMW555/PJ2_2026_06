# Neural Network and Deep Learning Project 2

This repository contains the code and experiment materials for Project 2 of the Neural Network and Deep Learning course.

## Project Goal

The project focuses on CIFAR-10 image classification and Batch Normalization analysis.

Main tasks include:

1. Train neural network models on CIFAR-10.
2. Build a network containing:

   * Fully connected layers
   * 2D convolutional layers
   * 2D pooling layers
   * Activation functions
3. Try at least one additional component, such as:

   * Batch Normalization
   * Dropout
   * Residual connection
4. Compare different model or training settings, including:

   * Number of filters or neurons
   * Loss functions or regularization
   * Activation functions
   * Optimizers
5. Compare VGG-A with and without Batch Normalization.
6. Generate visualizations such as training curves, loss landscape, or learned filters.

## Directory Structure

```text
.
├── codes/          # Source code for training, testing, and visualization
├── pic/            # Generated figures for the report
├── checkpoints/    # Trained model weights, ignored by git
├── data/           # CIFAR-10 dataset, ignored by git
├── results/        # Experiment logs and metrics
├── project_2_2026.pdf
├── AGENTS.md
└── README.md
```

## Environment

The code is expected to run with Python and PyTorch.

Recommended packages:

```bash
pip install torch torchvision matplotlib numpy
```

If additional packages are used later, they should be recorded here.

## How to Run

Run a quick smoke test comparing VGG-A with and without BatchNorm:

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --epochs 1 --n_items 1000
```

Use `--n_items` for a small CIFAR-10 subset during debugging. Omit it to use the
full training and test sets.

## Outputs

Generated outputs should be saved as follows:

```text
pic/              # figures used in the report
checkpoints/      # trained model weights
results/          # logs, metrics, and experiment records
```

## Notes

Large files such as datasets and model weights should not be committed to GitHub. They should be uploaded to a cloud drive or netdisk service, and the links should be included in the final report.
