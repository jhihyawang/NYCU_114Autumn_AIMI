# Lab3 - 肺部X光影像分類

本實驗使用深度學習模型對肺部X光影像進行四分類任務，包括正常、細菌性肺炎、病毒性肺炎和COVID-19。

## 📋 目錄

- [專案概述](#專案概述)
- [環境需求](#環境需求)
- [資料集結構](#資料集結構)
- [模型架構](#模型架構)
- [訓練設定](#訓練設定)
- [使用方式](#使用方式)
- [偽標籤策略](#偽標籤策略)
- [實驗結果](#實驗結果)

## 🎯 專案概述

本專案實作了一個完整的醫學影像分類流程，包含：

- **多種深度學習模型**：ResNet、EfficientNet、ConvNeXt等
- **多種損失函數**：Cross Entropy、Weighted CE、Focal Loss、Label Smoothing
- **資料增強技術**：肺部遮罩裁切、CLAHE對比度增強
- **偽標籤學習**：使用高信心預測擴充訓練集
- **完整訓練追蹤**：TensorBoard視覺化、混淆矩陣、分類報告

### 分類類別

- `normal`: 正常肺部
- `bacteria`: 細菌性肺炎
- `virus`: 病毒性肺炎
- `COVID-19`: 新冠肺炎

## 💻 環境需求

### Python版本

- Python >= 3.10

### 安裝依賴

使用 `uv` 套件管理器：

```bash
# 安裝所有依賴
uv sync
```

或使用 pip：

```bash
pip install torch torchvision albumentations opencv-python pandas scikit-learn \
    matplotlib seaborn tensorboard tqdm timm torch-optimizer
```

### 主要套件

- PyTorch >= 2.9.0
- torchvision >= 0.24.0
- albumentations >= 2.0.8
- opencv-python >= 4.12.0
- scikit-learn >= 1.7.2
- tensorboard >= 2.20.0

## 📁 資料集結構

```
Lab3/
├── no_exp/
│   ├── ori/           # 原始影像
│   ├── cropped/       # 肺部裁切影像
│   └── clahe/         # CLAHE處理影像
├── csv/
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data_sample.csv
└── ex4/
    ├── train.py       # 訓練主程式
    ├── model.py       # 模型定義
    ├── dataloader.py  # 資料載入器
    ├── loss.py        # 損失函數
    └── utils.py       # 工具函數
```

## 🏗️ 模型架構

支援的模型：

- **ResNet系列**: `resnet18`, `resnet34`, `resnet50`
- **EfficientNet系列**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- **ConvNeXt系列**: `convnext_tiny`, `convnext_small`

所有模型使用ImageNet預訓練權重進行初始化。

## ⚙️ 訓練設定

### 損失函數

- `ce`: Cross Entropy Loss
- `wce`: Weighted Cross Entropy (處理類別不平衡)
- `focal`: Focal Loss (聚焦困難樣本)
- `label_smooth`: Label Smoothing (防止過擬合)
- `weighted_label_smooth`: 結合權重與標籤平滑

### 優化器

- `adam`: Adam
- `adamw`: AdamW
- `sgd`: SGD with Momentum
- `rmsprop`: RMSprop

### 學習率調度器

- `cosine`: Cosine Annealing
- `step`: Step LR
- `reduce`: ReduceLROnPlateau
- `exponential`: Exponential LR
- `multistep`: MultiStep LR

## 🚀 使用方式

### 基本訓練

```bash
cd ex4

# 使用預設參數訓練
python train.py

# 指定模型和參數
python train.py \
    --model efficientnet_b2 \
    --dataset ../no_exp/cropped \
    --loss_type wce \
    --optimizer adamw \
    --lr 1e-4 \
    --batch_size 32 \
    --num_epochs 50 \
    --scheduler cosine \
    --use_amp \
    --experiment_id my_experiment
```

### 使用Shell腳本訓練

```bash
bash train.sh
```

### 主要參數說明

- `--model`: 模型名稱 (預設: `resnet18`)
- `--dataset`: 資料集路徑 (預設: `../no_exp/ori`)
- `--loss_type`: 損失函數類型 (預設: `ce`)
- `--optimizer`: 優化器類型 (預設: `adamw`)
- `--lr`: 學習率 (預設: `1e-4`)
- `--batch_size`: 批次大小 (預設: `32`)
- `--num_epochs`: 訓練輪數 (預設: `50`)
- `--scheduler`: 學習率調度器 (預設: `None`)
- `--patience`: 早停耐心值 (預設: `15`)
- `--use_amp`: 啟用混合精度訓練
- `--resize`: 影像大小 (預設: `512`)

### 推論測試集

```bash
cd ex4

python inference.py \
    --model efficientnet_b2 \
    --model_path result/my_experiment/weights/best.pt \
    --dataset ../no_exp/cropped \
    --output_dir result/my_experiment
```

## 🏷️ 偽標籤策略

本專案實作了偽標籤 (Pseudo-Labeling) 技術來擴充訓練資料：

### 流程

1. 使用訓練好的模型對測試集進行預測
2. 篩選高信心度的預測結果 (如 confidence >= 0.90)
3. 將高信心預測加入訓練集作為偽標籤
4. 使用擴充後的訓練集重新訓練模型

### 使用Jupyter Notebook

開啟 `Pseudo-Labeling.ipynb` 並依序執行：

1. 載入訓練好的模型
2. 對測試集進行預測並分析信心度分布
3. 選擇適當的信心度閾值
4. 生成偽標籤CSV檔案
5. 合併偽標籤到訓練集

### 信心度閾值建議

- `0.90`: 較寬鬆，可獲得更多偽標籤
- `0.95`: 平衡品質與數量
- `0.98`: 嚴格，僅保留極高信心預測
- `0.99`: 非常嚴格，適合特定類別

## 📊 實驗結果

訓練完成後，在 `ex4/result/{experiment_id}/` 目錄下會生成：

### 輸出檔案

- `weights/best.pt`: 最佳模型權重
- `accuracy_curve.png`: 訓練/驗證準確率曲線
- `f1_score_curve.png`: F1分數曲線
- `training_loss_curve.png`: 訓練損失曲線
- `confusion_matrix.png`: 混淆矩陣
- `test_predictions_{experiment_id}.csv`: 測試集預測結果
- `tensorboard/`: TensorBoard日誌檔案

### 查看TensorBoard

```bash
tensorboard --logdir ex4/result/{experiment_id}/tensorboard
```

### 評估指標

- **Accuracy**: 整體準確率
- **Macro F1-Score**: 各類別F1分數的平均值
- **Per-Class Metrics**: 每個類別的精確率、召回率、F1分數
- **Confusion Matrix**: 混淆矩陣視覺化

## 📈 常見實驗配置

### 配置 1: 基礎ResNet18

```bash
python train.py --model resnet18 --lr 1e-3 --batch_size 64
```

### 配置 2: EfficientNet + Weighted CE

```bash
python train.py \
    --model efficientnet_b2 \
    --loss_type wce \
    --dataset ../no_exp/cropped \
    --lr 1e-4 \
    --scheduler cosine \
    --use_amp
```

### 配置 3: ConvNeXt + Focal Loss

```bash
python train.py \
    --model convnext_small \
    --loss_type focal \
    --focal_gamma 2.0 \
    --optimizer adamw \
    --lr 5e-5 \
    --batch_size 16
```

### 配置 4: 使用偽標籤訓練

```bash
python train.py \
    --model efficientnet_b2 \
    --train_csv ../csv/train_data_with_pseudo.csv \
    --dataset ../no_exp/cropped \
    --loss_type wce \
    --lr 1e-4 \
    --use_amp
```

## 🔧 資料前處理

### 肺部遮罩裁切

使用 `lung-segmentation/` 中的U-Net模型進行肺部分割：

```bash
cd lung-segmentation
jupyter notebook preprocessing.ipynb
```

### CLAHE增強

使用 `ex4/preprocess.ipynb` 進行CLAHE對比度限制自適應直方圖均衡化。

## 📝 注意事項

1. **GPU記憶體**: 較大的模型(如EfficientNet-B2/B3、ConvNeXt)需要更多GPU記憶體，建議使用 `--use_amp` 啟用混合精度訓練
2. **類別不平衡**: COVID-19類別樣本較少(約1%)，建議使用 `wce` 或 `focal` 損失函數
3. **早停機制**: 預設耐心值為15 epochs，可透過 `--patience` 調整
4. **資料增強**: dataloader中已包含適當的資料增強策略
5. **偽標籤**: 使用偽標籤時需謹慎選擇信心度閾值，避免引入錯誤標籤
