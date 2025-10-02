import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自訂 Dataset
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if fpath.lower().endswith((".jpg", ".jpeg", ".png")):  # 避免非圖片檔
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# 資料前處理
def get_transform(train=True, resize=224, degree=10):
    if train:
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degree),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


# 建立 train / val / test dataloader
def get_dataloaders(data_dir="chest_xray", batch_size=32, resize=224, degree=10):
    train_dataset = ChestXrayDataset(
        root_dir=os.path.join(data_dir, "train"),
        transform=get_transform(train=True, resize=resize, degree=degree)
    )

    val_loader = None
    val_dir = os.path.join(data_dir, "val")
    if os.path.exists(val_dir):
        val_dataset = ChestXrayDataset(
            root_dir=val_dir,
            transform=get_transform(train=False, resize=resize)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ChestXrayDataset(
        root_dir=os.path.join(data_dir, "test"),
        transform=get_transform(train=False, resize=resize)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
