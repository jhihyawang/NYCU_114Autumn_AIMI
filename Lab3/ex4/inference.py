import warnings
from tqdm import tqdm
from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import cv2

from model import build_model
from dataloader import build_transforms, CLASS_NAMES
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from utils import plot_confusion_matrix

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# Validation Inference Function
# ============================================================
def inference_validation(model, device, val_csv, img_dir, img_size, output_dir, pred_weights=None):
    """
    對驗證集進行推論並保存結果
    
    Args:
        model: 訓練好的模型
        device: 設備 (cuda/cpu)
        val_csv: 驗證集 CSV 檔案路徑
        img_dir: 驗證集圖片目錄
        img_size: 圖片尺寸
        output_dir: 輸出目錄
        pred_weights: 可選的預測權重
    """
    print("\n" + "="*80)
    print("🔍 Starting Validation Set Inference")
    print("="*80)
    
    if pred_weights is not None:
        print(f"📊 Using prediction weights: {pred_weights.cpu().numpy()}")
    
    # 讀取驗證集 CSV
    df_val = pd.read_csv(val_csv)
    if "new_filename" not in df_val.columns:
        raise ValueError("❌ Validation CSV must contain 'new_filename' column")
    
    # 獲取真實標籤
    true_labels = df_val[CLASS_NAMES].values.argmax(axis=1)
    
    # 建立轉換
    transform = build_transforms(img_size)
    
    model.eval()
    all_preds = []
    all_probs = []
    results = []
    
    img_dir = Path(img_dir)
    
    for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc="🔍 Predicting"):
        fname = row["new_filename"]
        img_path = img_dir / fname
        
        if not img_path.exists():
            print(f"⚠️ Image not found: {img_path}")
            continue
        
        # 讀取並轉換圖片
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ Cannot read image: {img_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image)
        image_tensor = transformed["image"].unsqueeze(0).to(device)
        
        # 推論
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            
            # 應用預測權重（如果提供）
            if pred_weights is not None:
                weighted_probs = probs * pred_weights.to(device)
                pred = torch.argmax(weighted_probs, dim=1).item()
            else:
                pred = torch.argmax(probs, dim=1).item()
            
            all_preds.append(pred)
            all_probs.append(probs.cpu().numpy()[0])
        
        # 建立結果行
        result_row = {"new_filename": fname}
        for i, cls in enumerate(CLASS_NAMES):
            result_row[cls] = 1 if i == pred else 0
        results.append(result_row)
    
    # 轉換為 numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 計算評估指標
    print("\n" + "="*80)
    print("📊 Validation Set Evaluation Results")
    print("="*80)
    
    # 準確率
    accuracy = (all_preds == true_labels).sum() / len(true_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Macro F1-score
    f1_macro = f1_score(true_labels, all_preds, average='macro')
    print(f"Macro F1-score: {f1_macro:.4f}")
    
    # 分類報告
    report = classification_report(
        true_labels, all_preds, 
        target_names=CLASS_NAMES, 
        digits=4
    )
    print("\n📋 Classification Report:")
    print(report)
    
    # 混淆矩陣
    c_matrix = confusion_matrix(true_labels, all_preds)
    print("\n📊 Confusion Matrix:")
    print(c_matrix)
    
    # 保存結果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存預測結果 CSV
    output_csv = output_dir / "val_predictions.csv"
    df_out = pd.DataFrame(results, columns=["new_filename"] + CLASS_NAMES)
    df_out.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to: {output_csv}")
    
    # 2. 保存詳細結果（包含真實標籤和預測標籤）
    detailed_csv = output_dir / "val_predictions_detailed.csv"
    df_detailed = df_val.copy()
    df_detailed['predicted_class'] = [CLASS_NAMES[p] for p in all_preds]
    df_detailed['true_class'] = [CLASS_NAMES[t] for t in true_labels]
    df_detailed['correct'] = (all_preds == true_labels)
    
    # 加入每個類別的預測機率
    for i, cls in enumerate(CLASS_NAMES):
        df_detailed[f'prob_{cls}'] = all_probs[:, i]
    
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"✅ Detailed predictions saved to: {detailed_csv}")
    
    # 3. 保存分類報告
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("Validation Set Classification Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Macro F1-score: {f1_macro:.4f}\n\n")
        f.write("="*80 + "\n")
        f.write("Classification Report\n")
        f.write("="*80 + "\n\n")
        f.write(report)
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("Confusion Matrix\n")
        f.write("="*80 + "\n\n")
        f.write(str(c_matrix))
    print(f"✅ Classification report saved to: {report_path}")
    
    # 4. 保存混淆矩陣圖片
    plot_confusion_matrix(c_matrix, output_dir)
    
    # 5. 保存錯誤分析
    error_csv = output_dir / "error_analysis.csv"
    error_mask = (all_preds != true_labels)
    df_errors = df_detailed[error_mask].copy()
    df_errors = df_errors.sort_values('new_filename')
    df_errors.to_csv(error_csv, index=False)
    print(f"✅ Error analysis saved to: {error_csv}")
    print(f"   Total errors: {error_mask.sum()} / {len(true_labels)} ({error_mask.sum()/len(true_labels)*100:.2f}%)")
    
    # 6. 預測分佈統計
    print(f"\n📊 Prediction Distribution:")
    pred_dist = pd.Series(all_preds).value_counts().sort_index()
    true_dist = pd.Series(true_labels).value_counts().sort_index()
    
    print(f"\n{'Class':<15} {'True':<10} {'Predicted':<10} {'Difference':<10}")
    print("-" * 50)
    for i, cls in enumerate(CLASS_NAMES):
        true_count = true_dist.get(i, 0)
        pred_count = pred_dist.get(i, 0)
        diff = pred_count - true_count
        print(f"{cls:<15} {true_count:<10} {pred_count:<10} {diff:+<10}")
    
    print("="*80 + "\n")
    
    return accuracy, f1_macro, c_matrix, df_detailed

# ============================================================
# Main Function
# ============================================================
if __name__ == '__main__':
    parser = ArgumentParser(description='Inference script for validation set')
    
    # 基本設定
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model architecture name')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--resize', type=int, default=512,
                        help='Image resize dimension')
    
    # 資料集設定
    parser.add_argument('--dataset', type=str, default='../segmented_softclahe',
                        help='Dataset root directory')
    parser.add_argument('--val_csv', type=str, default='../csv/val_data.csv',
                        help='Validation CSV file path')
    
    # 輸出設定
    parser.add_argument('--output_dir', type=str, default='inference',
                        help='Output directory for inference results')
    
    # 預測權重設定
    parser.add_argument('--pred_weights', type=str, default=None,
                        help='Comma-separated prediction weights for each class')
    
    args = parser.parse_args()
    
    # 解析預測權重
    if args.pred_weights is not None:
        try:
            weights_list = [float(w.strip()) for w in args.pred_weights.split(',')]
            if len(weights_list) != args.num_classes:
                raise ValueError(f"Number of weights ({len(weights_list)}) must match num_classes ({args.num_classes})")
            args.pred_weights = torch.tensor(weights_list)
            print(f"📊 Prediction weights: {args.pred_weights.numpy()}")
        except Exception as e:
            raise ValueError(f"Error parsing pred_weights: {e}")
    else:
        args.pred_weights = None
    
    # 設定設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "="*80)
    print("🚀 Validation Set Inference")
    print("="*80)
    print(f"💻 Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"🏗️  Model: {args.model}")
    print(f"📦 Model weights: {args.model_path}")
    print(f"📁 Dataset: {args.dataset}")
    print(f"📄 Validation CSV: {args.val_csv}")
    print(f"💾 Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # 檢查模型檔案是否存在
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"❌ Model file not found: {args.model_path}")
    
    # 建立模型
    print("🏗️  Loading model...")
    model = build_model(model_name=args.model, pretrained=False, num_classes=args.num_classes)
    
    # 載入權重
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # 設定驗證集路徑
    val_img_dir = Path(args.dataset) / "val_images"
    if not val_img_dir.exists():
        raise FileNotFoundError(f"❌ Validation images directory not found: {val_img_dir}")
    
    # 執行推論
    accuracy, f1_macro, c_matrix, df_detailed = inference_validation(
        model=model,
        device=device,
        val_csv=args.val_csv,
        img_dir=val_img_dir,
        img_size=args.resize,
        output_dir=args.output_dir,
        pred_weights=args.pred_weights
    )
    
    print("\n" + "="*80)
    print("🎉 Inference Complete!")
    print("="*80)
    print(f"📊 Final Results:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Macro F1-score: {f1_macro:.4f}")
    print(f"\n📁 All results saved in: {Path(args.output_dir).resolve()}")
    print("="*80 + "\n")
