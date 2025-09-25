import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

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

def get_dataloaders(data_dir="chest_xray", batch_size=32, resize=224, degree=10):
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=get_transform(train=True, resize=resize, degree=degree)
    )

    val_loader = None
    val_dir = os.path.join(data_dir, "val")
    if os.path.exists(val_dir):
        val_dataset = datasets.ImageFolder(
            root=val_dir,
            transform=get_transform(train=False, resize=resize)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=get_transform(train=False, resize=resize)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader