# Project Instructions for Codex

This is a course project for Neural Network and Deep Learning.

Main goal:
- Complete Project 2 on CIFAR-10 and Batch Normalization.
- Prioritize correctness, reproducibility, clear experiments, and report-ready outputs.

Rules:
- Do not delete existing files unless explicitly asked.
- Keep code simple and suitable for undergraduate coursework.
- Prefer PyTorch implementation.
- Save all generated figures into ./pic.
- Save trained model weights into ./checkpoints, but do not commit large weight files.
- Add or update README.md when changing running commands.
- Make experiments reproducible by setting random seeds when possible.

Project requirements:
- Train neural networks on CIFAR-10.
- The network should include fully-connected layers, 2D convolutional layers, 2D pooling layers, and activations.
- Try at least one of BatchNorm, Dropout, Residual Connection, or other components.
- Compare different filters/neurons, losses or regularization, activations, and optimizers.
- Include model insight or visualization, such as filters, loss landscape, or interpretation.
- Compare VGG-A with and without BatchNorm.
- Generate figures for training curves and loss landscape comparison.