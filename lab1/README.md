1. 建立環境
curl -LsSf https://astral.sh/uv/install.sh | sh
重新開一個終端機
查是否安裝成功：uv --version

建立專案環境
uv venv lab1
source venv/bin/activate   # macOS / Linux
lab1\Scripts\activate      # Windows

Lab1/
│── dataloader.py   # 處理 dataset 與 transform
│── train.py        # 訓練流程 (main function)
│── inference.py    # 單張影像推論
│── models.py       # 模型定義 (ResNet18, ResNet50)
│── utils.py        # 工具函數 (metrics, plotting, confusion matrix)
│── README.md       # 說明文件