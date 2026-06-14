# Neural Network and Deep Learning Project 2

Course project on CIFAR-10 classification and Batch Normalization analysis.

The main model is a VGG-A style CNN with BatchNorm trained on CIFAR-10. The project also includes ablation experiments, loss-variation analysis, and first-layer filter visualization for the final report.

## Code Structure

```text
codes/VGG_BatchNorm/
  VGG_Loss_Landscape.py      Main script for smoke, BN comparison, landscape, ablation, and final runs
  visualize_filters.py       Visualizes first-layer convolution filters from a trained checkpoint
  models/vgg.py              VGG_A, VGG_A_BatchNorm, VGG_A_Light, VGG_A_Dropout
  data/loaders.py            CIFAR-10 dataloaders, subset training, and optional augmentation
  utils/nn.py                Weight initialization helpers
pic/                         Report figures
results/                     Local experiment metrics, ignored by Git
checkpoints/                 Local model weights, ignored by Git
```

## Environment

Recommended packages:

```bash
pip install torch torchvision matplotlib numpy tqdm python-docx
```

The code uses PyTorch and torchvision. GPU is used automatically when available.

## Dataset

CIFAR-10 is loaded with `torchvision.datasets.CIFAR10`.

Official source: https://www.cs.toronto.edu/~kriz/cifar.html

The dataset files are not included in GitHub. They can be downloaded automatically by the dataloader.

## Main Commands

BatchNorm comparison:

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --mode comparison --epochs 10 --n_items 10000 --batch_size 128 --lr 1e-3 --optimizer adam
```

CIFAR-10 ablation experiments:

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --mode ablation --epochs 5 --n_items 10000 --batch_size 128
```

Final regularized training:

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --mode final --model vgg_a_bn --epochs 30 --batch_size 128 --lr 1e-3 --optimizer adamw --weight_decay 5e-4 --scheduler cosine --augment --tag regularized
```

Filter visualization:

```bash
python codes/VGG_BatchNorm/visualize_filters.py --model vgg_a_bn --checkpoint checkpoints/final_vgg_a_bn_regularized.pth --output pic/first_layer_filters_regularized.png
```

Build the Word report draft:

```bash
python build_report_docx.py
```

## Main Result

Final regularized model:

- Model: `VGG_A_BatchNorm`
- Dataset: full CIFAR-10 training set
- Optimizer: AdamW
- Learning rate: `1e-3`
- Weight decay: `5e-4`
- Scheduler: cosine
- Augmentation: RandomCrop + RandomHorizontalFlip
- Best test accuracy: `0.8959`
- Best test error: `0.1041`
- Best epoch: `28`

The result is recorded in `results/final_training_regularized_results.json` locally. The `results/` directory is ignored by Git, so this README keeps the main summary.


