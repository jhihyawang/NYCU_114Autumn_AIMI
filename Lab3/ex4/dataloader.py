import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# ==========================================================
#  全域變數定義
# ==========================================================
CLASS_NAMES = ["normal", "bacteria", "virus", "COVID-19"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def gray_to_rgb(image, **kwargs):
    """Convert grayscale to RGB if needed"""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

# ==========================================================
#  Dataset for Train with Class-Specific Augmentation
# ==========================================================
class XrayCSVDataset(Dataset):
    def __init__(self, csv_path, img_dir, 
                 transform=None):
        """
        Args:
            csv_path: CSV 檔案路徑
            img_dir: 圖片目錄
            transform: 增強轉換
        """
        cv2.setNumThreads(0)
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.labels = self.df[CLASS_NAMES].values.argmax(axis=1)
        self.filenames = self.df["new_filename"].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.img_dir / self.filenames[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        transformed = self.transform(image=img)
        
        return transformed["image"], label

# ==========================================================
#  Dataset for Val (無增強)
# ==========================================================
class XrayCSVValDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        cv2.setNumThreads(0)
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.labels = self.df[CLASS_NAMES].values.argmax(axis=1)
        self.filenames = self.df["new_filename"].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.img_dir / self.filenames[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        transformed = self.transform(image=img)
        return transformed["image"], label

# ==========================================================
#  Dataset for Test
# ==========================================================
class XrayTestDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        cv2.setNumThreads(0)
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.filenames = self.df["new_filename"].values
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = self.img_dir / self.filenames[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image=img) if self.transform else {"image": img}
        return transformed["image"], Path(self.filenames[idx]).name

# ==========================================================
#  Transforms
# ==========================================================
def build_transforms(img_size=224):
    """
    基礎轉換: 僅用於驗證集和測試集 (無增強)
    """
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        
        # 灰階轉 RGB
        A.Lambda(
            name="GrayToRGB",
            image=gray_to_rgb,
            p=1.0
        ),
        
        # ImageNet 標準化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # 轉換為 PyTorch 張量
        ToTensorV2()
    ])
    return base_transform

def build_augment_transforms(img_size=224):
    transform = A.Compose([
        # --- 基礎預處理 ---
        A.Resize(img_size, img_size),

        # --- 幾何增強 (Geometric Augmentations) ---
        # 這是模擬病人拍攝角度、位置和距離的關鍵
        A.HorizontalFlip(p=0.5),
        
        # A.Affine 整合了旋轉、平移、縮放，非常強大
        # 旋轉 -10 到 +10 度
        # 平移 10%
        # 縮放 90% 到 110%
        A.Affine(
            rotate=(-10, 10), 
            translate_percent=0.1, 
            scale=(0.9, 1.1), 
            p=0.7
        ),

        # --- 像素/醫學影像增強 (Pixel/Medical Augmentations) ---
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.7),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.5),
        ], p=0.8), # 80% 的機率從以上三種中選一種執行

        
        # --- 遮擋/丟棄 (Occlusion/Dropout) ---
        # 在影像上隨機挖 4-8 個黑洞
        A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(int(img_size * 0.05), int(img_size * 0.1)),
            hole_width_range=(int(img_size * 0.05), int(img_size * 0.1)),
            p=0.5
        ),

        # --- 最終轉換 (Finalization) ---
        A.Lambda(
            name="GrayToRGB",
            image=gray_to_rgb,
            p=1.0  # 總是執行
        ),
        
        # ImageNet 標準化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # 轉換為 PyTorch 張量
        ToTensorV2()
    ])
    return transform

# ==========================================================
#  計算類別權重 (根據論文公式)
# ==========================================================
def calculate_class_weights(csv_path):
    """
    根據論文公式計算類別權重:
    ωᵢ = (總樣本數) / (類別數 × 第i類的樣本數)
    """
    df = pd.read_csv(csv_path)
    labels = df[CLASS_NAMES].values.argmax(axis=1)
    
    total_samples = len(labels)
    num_classes = len(CLASS_NAMES)
    
    # 計算每個類別的樣本數
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # 處理 0-count 的情況，避免除以零
    class_counts = np.where(class_counts == 0, 1, class_counts)
    
    # 根據論文公式計算權重
    class_weights = total_samples / (num_classes * class_counts)
    
    # 轉換為 torch tensor
    weights = torch.FloatTensor(class_weights)
    
    print(f"\n{'='*60}")
    print(f"{'類別權重計算':^60}")
    print(f"{'='*60}")
    print(f"總樣本數: {total_samples}")
    print(f"類別數: {num_classes}")
    print(f"{'-'*60}")
    for i, class_name in enumerate(CLASS_NAMES):
        percentage = (class_counts[i] / total_samples) * 100
        print(f"{class_name:12s} | 樣本數: {class_counts[i]:4d} ({percentage:5.2f}%) | 權重: {weights[i]:.4f}")
    print(f"{'='*60}\n")
    
    return weights

# ==========================================================
#  DataLoader Builders
# ==========================================================
def get_train_val_loaders(data_root="datasets", img_size=224, batch_size=32, num_workers=4, train_csv="../csv/train_data.csv"):
    """
    建立訓練和驗證的 DataLoader
    """
    # train_csv = "../csv/train_data.csv"
    train_csv = train_csv
    val_csv = "../csv/val_data.csv"
    
    # 影像目錄路徑
    train_img_dir = Path(data_root) / "train_images"
    val_img_dir = Path(data_root) / "val_images"
    
    # 建立不同強度的轉換
    base_transform = build_transforms(img_size)  # 驗證集用
    transform = build_augment_transforms(img_size) 

    # 建立 Dataset
    train_ds = XrayCSVDataset(
        train_csv, 
        train_img_dir,
        transform=transform,
    )
    
    val_ds = XrayCSVValDataset(val_csv, val_img_dir, transform=base_transform)
    
    # 計算類別權重
    class_weights = calculate_class_weights(train_csv)
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, class_weights

def get_test_loader(data_root="datasets", img_size=224, batch_size=32, num_workers=4):
    """
    建立測試集的 DataLoader
    """
    test_csv = "../csv/test_data_sample.csv"
    test_img_dir = Path(data_root) / "test_images"
    
    base_transform = build_transforms(img_size)
    
    test_ds = XrayTestDataset(test_csv, test_img_dir, transform=base_transform)
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True if num_workers > 0 else False
    )
    
    return test_loader

# ==========================================================
#  混淆矩陣分析工具
# ==========================================================
def analyze_confusion(y_true, y_pred):
    """
    分析混淆情況，特別關注 bacteria 和 virus 之間的混淆
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{'混淆矩陣分析':^60}")
    print(f"{'='*60}")
    
    # 顯示完整混淆矩陣
    print("\n混淆矩陣:")
    print(f"{'':12s} ", end="")
    for name in CLASS_NAMES:
        print(f"{name:12s}", end=" ")
    print()
    
    for i, true_name in enumerate(CLASS_NAMES):
        print(f"{true_name:12s} ", end="")
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i, j]:12d}", end=" ")
        print()
    
    # 特別分析 bacteria-virus 混淆
    bacteria_idx = CLASS_TO_IDX["bacteria"]
    virus_idx = CLASS_TO_IDX["virus"]
    
    print(f"\n{'-'*60}")
    print(f"❗ 關鍵混淆分析 (bacteria ↔ virus)")
    print(f"{'-'*60}")
    print(f"Bacteria → Virus 誤判: {cm[bacteria_idx, virus_idx]:4d} 次")
    print(f"Virus → Bacteria 誤判: {cm[virus_idx, bacteria_idx]:4d} 次")
    print(f"總混淆次數: {cm[bacteria_idx, virus_idx] + cm[virus_idx, bacteria_idx]:4d} 次")
    
    # 計算混淆率
    bacteria_total = cm[bacteria_idx].sum()
    virus_total = cm[virus_idx].sum()
    bacteria_to_virus_rate = (cm[bacteria_idx, virus_idx] / bacteria_total * 100) if bacteria_total > 0 else 0
    virus_to_bacteria_rate = (cm[virus_idx, bacteria_idx] / virus_total * 100) if virus_total > 0 else 0
    
    print(f"Bacteria → Virus 誤判率: {bacteria_to_virus_rate:.2f}%")
    print(f"Virus → Bacteria 誤判率: {virus_to_bacteria_rate:.2f}%")
    
    print(f"{'='*60}\n")
    
    # 顯示詳細分類報告
    print("分類報告:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    
    return cm