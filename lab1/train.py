import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from argparse import ArgumentParser
from tqdm import tqdm

from dataloader import get_dataloaders
from utils import measurement, plot_accuracy, plot_f1_score


def test(loader, model, device):
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        model.eval()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = torch.max(model(imgs), 1).indices
            sub_tp, sub_tn, sub_fp, sub_fn = measurement(preds, labels)
            tp += sub_tp; tn += sub_tn; fp += sub_fp; fn += sub_fn
    acc = (tp + tn) / (tp + tn + fp + fn) * 100
    f1 = (2 * tp) / (2 * tp + fp + fn)
    c_matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]
    return acc, f1, c_matrix


def train(args, device, train_loader, val_loader, model, criterion, optimizer):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list, val_acc_list, f1_list = [], [], []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        tp, tn, fp, fn = 0, 0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.max(outputs, 1).indices
            sub_tp, sub_tn, sub_fp, sub_fn = measurement(preds, labels)
            tp += sub_tp; tn += sub_tn; fp += sub_fp; fn += sub_fn

        train_acc = (tp + tn) / (tp + tn + fp + fn) * 100
        val_acc, f1, _ = test(val_loader, model, device)

        print(f"Epoch {epoch} | Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | F1 {f1:.4f}")

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        f1_list.append(f1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, f"{args.model}_best.pt")

    return train_acc_list, val_acc_list, f1_list


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", default="chest_xray")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"## Using device: {device} ##")

    # dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.dataset, batch_size=args.batch_size, resize=224, degree=10
    )

    # model
    model = models.resnet18(pretrained=True) if args.model == "resnet18" else models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_acc, val_acc, f1_list = train(args, device, train_loader, val_loader, model, criterion, optimizer)

    # plots
    plot_accuracy(train_acc, val_acc)
    plot_f1_score(f1_list)
