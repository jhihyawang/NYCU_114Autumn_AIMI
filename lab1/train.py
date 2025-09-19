import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataloader import get_dataloaders
from models import get_model
from utils import calculate_accuracy, calculate_f1, plot_metrics

def train(train_dir, test_dir, val_dir=None, model_name="resnet18", epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(train_dir, test_dir, val_dir, batch_size)

    # model
    model = get_model(model_name, num_classes=2).to(device)

    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc, train_f1 = [], []
    val_acc, val_f1 = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            running_loss += loss.item()

        acc = calculate_accuracy(np.array(y_true), np.array(y_pred))
        f1 = calculate_f1(np.array(y_true), np.array(y_pred))
        train_acc.append(acc)
        train_f1.append(f1)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Acc: {acc:.4f} - F1: {f1:.4f}")

        # (可加驗證 val_loader)

    plot_metrics(train_acc, train_f1, val_acc, val_f1)

    # 存模型
    torch.save(model.state_dict(), f"{model_name}_best.pth")

if __name__ == "__main__":
    train("chest_xray/train", "chest_xray/test", "chest_xray/val", model_name="resnet18", epochs=5)
