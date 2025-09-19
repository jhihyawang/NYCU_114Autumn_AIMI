import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import get_model
from utils import calculate_accuracy, calculate_f1, plot_confusion_matrix

def inference(model_path, test_dir, model_name="resnet18"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = get_model(model_name, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = calculate_accuracy(np.array(y_true), np.array(y_pred))
    f1 = calculate_f1(np.array(y_true), np.array(y_pred))
    print(f"Test Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    plot_confusion_matrix(y_true, y_pred, class_names=test_dataset.classes)

if __name__ == "__main__":
    inference("resnet18_best.pth", "chest_xray/test", model_name="resnet18")
