# Lab1 : Pneumonia Classification from Chest X-ray Images

This lab implements a deep learning solution for classifying pneumonia from chest X-ray images using ResNet architectures and other models.

## 📋 Project Overview

- **Objective**: Binary classification of chest X-ray images (Normal vs Pneumonia)
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Models Implemented**: ResNet18, ResNet50, DenseNet121, Vision Transformer (vit_base_patch16_224)

## 🚀 Environment Setup

This project uses `uv` for Python environment management.

### Prerequisites
- Python 3.10.12
- CUDA-capable GPU (recommended)
- `uv` package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/jhihyawang/NYCU_114Autumn_AIMI.git
cd NYCU_114Autumn_AIMI/Lab1
```

2. **Create virtual environment with uv**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
uv sync
```

## 📁 Project Structure

```
.
├── train.py              # Training script
├── inference.py          # Inference and evaluation script
├── models.py             # Model architectures
├── dataloader.py         # Custom dataset and dataloader
├── utils.py              # Utility functions (metrics, plotting)
├── chest_xray/           # Dataset directory
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── weight/               # Saved model weights
├── result/               # Training results and plots
└── pyproject.toml        # dependencies
```

## 📊 Dataset Preparation

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract and place in the project directory as `chest_xray/`
3. The dataset structure should match the format shown above

## 🎯 Training

### Basic Training
```bash
uv run code/train.py
```

### Training with Custom Parameters
```bash
uv run code/train.py \
    --model resnet50 \
    --num_epochs 30 \
    --batch_size 128 \
    --lr 1e-5 \
    --wd 0.9 \
    --patience 6 \
    --resize 224 \
    --degree 10
```

### Available Arguments
- `--model`: Model architecture (resnet18, resnet50, densenet121, vit_base_patch16_224)
- `--num_epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-5)
- `--wd`: Weight decay (default: 0.9)
- `--patience`: Early stopping patience (default: 6)
- `--resize`: Image resize dimension (default: 224)
- `--degree`: Rotation degree for data augmentation (default: 10)


## 🔍 Inference

### Using Trained Weights

If you want to use the trained model weights without training from scratch:

1. **Download the pretrained weights** from Google Drive:
   - [Download Model Weights (All 4 Models)](https://drive.google.com/file/d/1_mWGctchyNjOzfA4U_mweIOWIIzl0LiO/view?usp=sharing)
   - Includes: `resnet18_best.pt`, `resnet50_best.pt`, `vit_base_patch16_224_best.pt`, `densenet121_best.pt`

2. **Place the weights in the correct directory**:
   ```bash
   mkdir -p weight
   # Extract and move all .pt files to the weight folder
   # Or move specific model weights you want to use
   ```

3. **Run inference** (choose your model):
   ```bash
   # For ResNet50 (Best performance: 91.67%)
   python inference.py --model resnet50
   
   # For ResNet18 (91.51%)
   python inference.py --model resnet18
   
   # For Vision Transformer (90.71%)
   python inference.py --model vit_base_patch16_224
   
   # For DenseNet121 (87.82%)
   python inference.py --model densenet121
   ```

### Training Your Own Model

After training your model using `train.py`, evaluate it on the test set:

```bash
python inference.py --model <model_name>
```

Make sure the model name matches the one you trained.

## 📈 Results

### Model Performance Summary

| Model | Test Accuracy | F1-Score | Recall | Precision |
|-------|--------------|----------|---------|-----------|
| **ResNet50** | **91.67%** | **0.9364** | **0.9821** | 0.8949 |
| ResNet18 | 91.51% | 0.9351 | 0.9795 | 0.8946 |
| ViT-Base | 90.71% | 0.9293 | 0.9769 | 0.8860 |
| DenseNet121 | 87.82% | 0.9102 | **0.9872** | 0.8443 |

**Best Performing Model**: ResNet50 (91.67% accuracy, 0.9364 F1-score)

The training process generates the following outputs in `result/<model_name>/`:

1. **accuracy_curve.png**: Training and validation accuracy over epochs
2. **f1_score_curve.png**: F1-score over epochs
3. **confusion_matrix.png**: Confusion matrix of the final results

Model weights are saved in `weight/<model_name>_best.pt`

## 🎨 Data Augmentation

The custom dataloader includes:
- Random resized crop (scale: 0.8-1.0)
- Random rotation (default: ±10°)
- Random horizontal flip (p=0.3)
- Color jitter (brightness=0.05, contrast=0.05)
- Random erasing (p=0.1)
- ImageNet normalization

## 🏆 Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall

### Class Imbalance Handling
The training uses weighted CrossEntropyLoss with class weights [3.8896346, 1.346] to handle dataset imbalance.

## 💾 Model Checkpoint

The best model is saved based on the lowest validation loss with early stopping mechanism. The checkpoint includes the complete model state dictionary.

## 🛠️ Technical Details

- **Framework**: PyTorch
- **Pretrained Weights**: ImageNet pretrained models
- **Optimizer**: Adam
- **Loss Function**: Weighted Cross-Entropy Loss
- **Early Stopping**: Patience of 6 epochs (configurable)

## 📝 Notes

- Random seed is set to 39 for reproducibility
- The model automatically uses GPU if available

## 👤 Author
 
**Name**: Jhihya Wang

**Course**: NYCU 2025 Autumn AIMI

## 📧 Contact

For questions or issues, please open an issue in this repository.

---

**Last Updated**: October 2025
