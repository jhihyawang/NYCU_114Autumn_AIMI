1. 建立環境
curl -LsSf https://astral.sh/uv/install.sh | sh
重新開一個終端機
查是否安裝成功：uv --version

建立專案環境
uv venv lab1
source venv/bin/activate   # macOS / Linux
lab1\Scripts\activate      # Windows