#!/bin/bash

# Validation Set Inference Script
# 使用說明：./run_inference.sh <model_path> [options]

# 預設參數
MODEL="efficientnet_b2"
RESIZE=448
DATASET="../no_exp/ori"
VAL_CSV="../csv/val_data.csv"
OUTPUT_DIR="inference"
PRED_WEIGHTS="0.6,0.6,0.6,1.0"

# 檢查是否提供模型路徑
if [ -z "$1" ]; then
    echo "❌ Error: Model path is required"
    echo ""
    echo "Usage: ./run_inference.sh <model_path> [options]"
    echo ""
    echo "Example:"
    echo "  ./run_inference.sh results/my_experiment/weights/best.pt"
    echo ""
    echo "Options:"
    echo "  --model MODEL           Model architecture (default: resnet18)"
    echo "  --resize SIZE           Image resize dimension (default: 512)"
    echo "  --dataset PATH          Dataset root directory (default: ../segmented_softclahe)"
    echo "  --val_csv PATH          Validation CSV file (default: ../csv/val_data.csv)"
    echo "  --output_dir PATH       Output directory (default: inference)"
    echo "  --pred_weights WEIGHTS  Prediction weights (e.g., '1.0,1.0,0.7,1.0')"
    echo ""
    exit 1
fi

MODEL_PATH="$1"
shift

# 解析其他參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --resize)
            RESIZE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --val_csv)
            VAL_CSV="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --pred_weights)
            PRED_WEIGHTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 檢查模型檔案是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# 顯示執行資訊
echo "================================================"
echo "🚀 Running Validation Set Inference"
echo "================================================"
echo "📦 Model Path:     $MODEL_PATH"
echo "🏗️  Model:          $MODEL"
echo "📏 Resize:         $RESIZE"
echo "📁 Dataset:        $DATASET"
echo "📄 Val CSV:        $VAL_CSV"
echo "💾 Output Dir:     $OUTPUT_DIR"
if [ -n "$PRED_WEIGHTS" ]; then
    echo "📊 Pred Weights:   $PRED_WEIGHTS"
fi
echo "================================================"
echo ""

# 建立推論指令
CMD="python inference.py \
    --model_path \"$MODEL_PATH\" \
    --model $MODEL \
    --resize $RESIZE \
    --dataset \"$DATASET\" \
    --val_csv \"$VAL_CSV\" \
    --output_dir \"$OUTPUT_DIR\""

if [ -n "$PRED_WEIGHTS" ]; then
    CMD="$CMD --pred_weights \"$PRED_WEIGHTS\""
fi

# 執行推論
eval $CMD

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ Inference completed successfully!"
    echo "📁 Results saved in: $OUTPUT_DIR/"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "❌ Inference failed!"
    echo "================================================"
    exit 1
fi
