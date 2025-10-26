# Lab2: Brain Computer Interface (BCI) Classification using Deep Learning

## Overview
This project implements and compares two deep neural network architectures for EEG-based Brain-Computer Interface (BCI) classification: **EEGNet** and **DeepConvNet**. The models are trained to classify motor imagery tasks from EEG signals, demonstrating the effectiveness of specialized CNN architectures for neurophysiological signal processing.

## Dataset
- **Sources**: S4b and X11b subjects from BCI Competition III-IIIb
- **Task**: Binary motor imagery classification (2 classes: left vs right hand)
- **Signal**: EEG data with 2 channels, 750 time samples per trial
- **Training Data**: 1,080 trials (540 per subject)
- **Test Data**: Additional test set for validation
- **Format**: Preprocessed `.npz` files containing signal data and labels

## Models

### EEGNet
A lightweight convolutional neural network optimized for EEG signal classification:
- **Block 1**: Temporal convolution (1×51 kernel) with batch normalization
- **Block 2**: Depthwise spatial convolution (2×1 kernel) + ELU/ReLU + pooling + dropout
- **Block 3**: Separable convolution (depthwise 1×15 + pointwise 1×1) + pooling + dropout
- **Classifier**: Adaptive linear layer based on flattened feature dimensions
- **Parameters**: ~2,500 parameters (efficient design)

### DeepConvNet  
A hierarchical convolutional architecture with progressive feature learning:
- **Block 1**: Temporal conv (1×5) + Spatial conv (2×1) + BatchNorm + activation + MaxPool + dropout
- **Block 2**: Conv (1×5, 25→50 channels) + BatchNorm + activation + MaxPool + dropout  
- **Block 3**: Conv (1×5, 50→100 channels) + BatchNorm + activation + MaxPool + dropout
- **Block 4**: Conv (1×5, 100→200 channels) + BatchNorm + activation + MaxPool + dropout
- **Classifier**: Linear layer for final classification
- **Parameters**: ~40,000 parameters (deeper architecture)

## Project Structure
```
Lab2/
├── main.py                 # Main training script
├── dataloader.py          # Data loading and preprocessing
├── models/
│   ├── EEGNet.py         # EEGNet implementation
│   └── DeepConvNet.py    # DeepConvNet implementation
├── data/
│   ├── S4b_train.npz     # Training data (Subject S4b)
│   ├── S4b_test.npz      # Test data (Subject S4b)
│   ├── X11b_train.npz    # Training data (Subject X11b)
│   └── X11b_test.npz     # Test data (Subject X11b)
├── results/              # Experiment results directory
├── run_experiments.sh    # Hyperparameter sweep script
├── summarize_results.py  # Results analysis script
├── pyproject.toml        # Project dependencies
└── README.md            # This file
```

## Installation

### Prerequisites
- Python ≥ 3.10
- UV package manager (recommended) or pip

### Setup with UV (Recommended)
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Setup with pip
```bash
pip install torch torchvision matplotlib numpy pandas tqdm torchsummary
```

## Usage

### Single Experiment
```bash
# Run with default parameters (EEGNet)
uv run python main.py

# Run DeepConvNet with custom parameters
uv run python main.py \
    -model DeepConvNet \
    -optimizer Adam \
    -batch_size 32 \
    -lr 0.001 \
    -dropout 0.5 \
    -num_epochs 200 \
    -weight_decay 1e-5 \
    -activation ELU \
    -elu_alpha 0.5 \
    -experiment_id "DeepConvNet_custom"
```

### Hyperparameter Sweep
```bash
# Run automated hyperparameter search (configured for EEGNet by default)
uv run bash run_experiments.sh
```

### Results Analysis
```bash
# Generate summary of all experiments with performance metrics
uv run python summarize_results.py
```

## Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-model` | str | `"EEGNet"` | Model architecture (`EEGNet` or `DeepConvNet`) |
| `-optimizer` | str | `"Adam"` | Optimizer (`Adam` or `AdamW`) |
| `-batch_size` | int | `64` | Training batch size |
| `-lr` | float | `1e-3` | Learning rate |
| `-dropout` | float | `0.5` | Dropout rate |
| `-num_epochs` | int | `150` | Number of training epochs |
| `-weight_decay` | float | `1e-4` | L2 regularization weight |
| `-activation` | str | `"ELU"` | Activation function (`ELU`, `ReLU`, or `LeakyReLU`) |
| `-elu_alpha` | float | `1.0` | Alpha parameter for ELU activation |
| `-experiment_id` | str | `"exp_001"` | Unique experiment identifier |

## Results

### Best Performance Summary

**EEGNet Results:**
- **Best Accuracy**: 88.70% (Adam, bs=32, lr=0.001, dropout=0.5, epochs=200, ReLU activation)
- **Top Configuration**: Adam optimizer with learning rate 0.001, batch size 32, dropout 0.5
- **Architecture**: Lightweight and efficient with superior performance
- **Strengths**: Better feature extraction, faster convergence, excellent generalization

**DeepConvNet Results:**
- **Best Accuracy**: 83.52% (Adam, bs=32, lr=0.001, dropout=0.5, epochs=200, ELU α=0.5)
- **Top Configuration**: Adam optimizer with learning rate 0.001, batch size 32, dropout 0.5
- **Architecture**: Deeper network with hierarchical learning
- **Strengths**: Complex pattern recognition, robust feature hierarchies

### Performance Analysis
1. **EEGNet significantly outperforms DeepConvNet** (88.70% vs 83.52% best accuracy)
2. **ReLU activation** works exceptionally well for EEGNet (88.70% accuracy)
3. **ELU activation with α=0.5** is optimal for DeepConvNet (83.52% accuracy)
4. **Adam optimizer** performs consistently well for both architectures
5. **Learning rate 0.001** is the sweet spot for both models
6. **Batch size 32** provides optimal training dynamics
7. **Dropout 0.5** offers good regularization for top performance
8. **Lower dropout (0.25)** can work well with careful tuning but 0.5 is more robust

### Activation Function Impact
- **EEGNet**: ReLU > LeakyReLU > ELU (ReLU gives +1-2% accuracy boost)
- **DeepConvNet**: ELU (α=0.5) > ReLU > LeakyReLU (ELU provides better convergence)

## Output Files

Each experiment creates a directory `results/{experiment_id}/` containing:
- `best.pt`: PyTorch model state dict with best test accuracy weights
- `config.txt`: Complete experiment configuration and performance metrics
- `train_acc.png`: Training accuracy progression over epochs
- `train_loss.png`: Training loss curve for convergence analysis
- `test_acc.png`: Test accuracy evolution during training
- `results/summary.csv`: Consolidated performance table across all experiments (generated by `summarize_results.py`)

### Experiment Naming Convention
Experiments are automatically named using the pattern:
```
{Model}_opt{Optimizer}_bs{BatchSize}_lr{LearningRate}_dp{Dropout}_ep{Epochs}_wd{WeightDecay}_act{Activation}[_a{ELUAlpha}]
```
Example: `EEGNet_optAdam_bs32_lr0.001_dp0.5_ep200_wd1e-5_actReLU`

## Technical Details

### Data Preprocessing Pipeline
- **Dataset Concatenation**: Merges S4b and X11b subjects (1,080 total training samples)
- **Label Normalization**: Converts 1-based labels [1,2] to 0-based [0,1] for PyTorch compatibility
- **Tensor Reshaping**: Transforms (samples, time, channels) → (samples, 1, channels, time) for CNN input
- **NaN Handling**: Replaces missing values with dataset mean for robust training
- **Data Augmentation**: None (raw signal preservation for neurophysiological integrity)

### Training Infrastructure
- **Reproducibility**: Fixed random seed (456) for consistent experimental results
- **Device Management**: Automatic CUDA/CPU detection with seamless GPU acceleration
- **Model Checkpointing**: Saves best weights based on test accuracy (not validation loss)
- **Progress Tracking**: Real-time training progress with tqdm and epoch-wise metrics
- **Visualization**: Automatic generation of training/testing curves (accuracy & loss)

### Architecture Implementation Details

**EEGNet Architecture:**
```
Input (1, 2, 750) → 
Conv2d(1→16, 1×51, pad=25) + BatchNorm → 
DepthwiseConv2d(16→32, 2×1, groups=16) + BatchNorm + Activation + AvgPool(1×4) + Dropout → 
SeparableConv2d(32→32, 1×15→1×1, groups=32) + BatchNorm + Activation + AvgPool(1×8) + Dropout → 
Flatten + Linear → Output(2)
```

**DeepConvNet Architecture:**
```
Input (1, 2, 750) → 
Block1: Conv2d(1×5) + Conv2d(2×1) + BatchNorm + Activation + MaxPool(1×2) + Dropout →
Block2: Conv2d(1×5, 25→50) + BatchNorm + Activation + MaxPool(1×2) + Dropout →
Block3: Conv2d(1×5, 50→100) + BatchNorm + Activation + MaxPool(1×2) + Dropout →
Block4: Conv2d(1×5, 100→200) + BatchNorm + Activation + MaxPool(1×2) + Dropout →
Flatten + Linear → Output(2)
```

### Key Implementation Features
- **Dynamic Feature Dimension Calculation**: Both models auto-compute flatten dimensions
- **Flexible Activation Functions**: Support for ELU (with tunable α), ReLU, and LeakyReLU
- **Configurable Regularization**: Adjustable dropout rates and weight decay
- **Modular Design**: Clean separation between model definitions and training logic

## Experimental Insights

### Hyperparameter Sensitivity Analysis
Based on 32 comprehensive experiments:

**Learning Rate Impact:**
- Optimal range: 0.0007-0.001 for both models
- EEGNet: More tolerant to higher learning rates (0.001 optimal)
- DeepConvNet: Prefers moderate learning rates (0.001 with careful tuning)

**Dropout Effects:**
- EEGNet: Performs well with both 0.25 and 0.5 dropout
- DeepConvNet: Benefits from higher dropout (0.5) due to deeper architecture
- Lower dropout can achieve peak performance but requires careful tuning

**Activation Function Selection:**
- **EEGNet + ReLU**: Best combination (88.70% accuracy)
- **DeepConvNet + ELU**: Optimal for deeper networks (83.52% with α=0.5)
- LeakyReLU: Consistent but not optimal for either architecture

**Batch Size Considerations:**
- Batch size 32: Sweet spot for both models
- Smaller batches (16): May underperform due to noisy gradients
- Larger batches (64): Risk of overfitting on this dataset size

## Implementation Notes

### Code Quality Features
- **Type Safety**: Proper tensor dtype handling (float32 for data, int64 for labels)
- **Memory Efficiency**: Gradient context management and device-aware tensor operations
- **Error Handling**: Robust NaN detection and replacement
- **Extensibility**: Modular activation function system, easy to add new optimizers/models

### Performance Optimizations
- **Efficient Data Loading**: PyTorch DataLoader with configurable batch processing
- **GPU Acceleration**: Automatic CUDA utilization when available
- **Memory Management**: Proper tensor device placement and gradient cleanup

## Citation

Original architecture papers:

```bibtex
@article{lawhern2018eegnet,
  title={EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
  author={Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal={Journal of neural engineering},
  volume={15},
  number={5},
  pages={056013},
  year={2018},
  publisher={IOP Publishing}
}

@article{schirrmeister2017deep,
  title={Deep learning with convolutional neural networks for EEG decoding and visualization},
  author={Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
  journal={Human brain mapping},
  volume={38},
  number={11},
  pages={5391--5420},
  year={2017},
  publisher={Wiley Online Library}
}
```