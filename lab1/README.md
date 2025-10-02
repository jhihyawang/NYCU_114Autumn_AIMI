好的 👍 這裡給你一份 **README.md 的 Markdown 版本**，可以直接存成 `README.md` 使用：

```markdown
# Lab1 - Pneumonia Classification (Chest X-ray)

## 📌 專案目標
本實驗使用 Kaggle Chest X-ray Pneumonia dataset，利用 **ResNet18 / ResNet50** 做二分類任務 (Normal vs Pneumonia)。  
主要完成以下目標：
1. 撰寫自訂 DataLoader
2. 訓練 ResNet 模型
3. 在 validation/test dataset 上進行評估
4. 繪製 Accuracy、F1-score 曲線
5. 繪製最終 Confusion Matrix

---

## 📂 專案結構
```

.
├── dataloader.py        # 自訂 ChestXrayDataset + get_dataloaders()
├── utils.py             # metrics & 繪圖函式
├── train.py             # 訓練程式
├── inference.py         # 測試程式
└── README.md

```

---

## 📊 Dataset
請先下載 Kaggle dataset:  
👉 [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

並解壓縮至以下結構：
```

chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/

````

---

## ⚙️ 環境需求
- Python 3.8+
- PyTorch 1.10+
- torchvision
- matplotlib
- seaborn
- tqdm

安裝方式：
```bash
pip install torch torchvision matplotlib seaborn tqdm
````

---

## 🚀 使用方式

### 1. 訓練模型

以 ResNet18 為例：

```bash
python train.py --model resnet18 --num_epochs 20 --batch_size 32
```

以 ResNet50 為例：

```bash
python train.py --model resnet50 --num_epochs 20 --batch_size 32
```

訓練完成後，會輸出：

* `resnet18_best.pt` / `resnet50_best.pt` (最佳模型權重)
* `accuracy_curve.png` (Train vs Val Accuracy 曲線)
* `f1_score_curve.png` (Validation F1-score 曲線)

---

### 2. 測試模型

載入最佳模型，並在 **test dataset** 做最終評估：

```bash
python inference.py --model resnet18
```

輸出：

* 終端機顯示：Test Accuracy & F1-score
* `confusion_matrix.png` (測試結果混淆矩陣)

---

## 📈 實驗結果 (範例)

* **ResNet18 (20 epoch)**

  * Validation Acc: 100.00%
  * Validation F1: 1.0000
  * Test Acc: 90.87%
  * Test F1: 0.9311

輸出圖表：

* ![Accuracy Curve](accuracy_curve.png)
* ![F1 Curve](f1_score_curve.png)
* ![Confusion Matrix](confusion_matrix.png)

---

## 📝 注意事項

* Validation dataset 為可選，但使用後可幫助挑選最佳模型 (Bonus 分數)
* Test dataset 僅能在最終評估使用，不可用於訓練或模型選擇
* 請在報告內附上 **結果截圖** 及 **討論分析**