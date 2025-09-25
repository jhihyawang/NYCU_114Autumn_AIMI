import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from tqdm import tqdm

from dataloader import get_dataloaders
from models import get_model
from utils import measurement, plot_accuracy, plot_f1_score, plot_confusion_matrix


def test(loader, model, device):
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        model.eval()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices
            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp; tn += sub_tn; fp += sub_fp; fn += sub_fn

    c_matrix = [[int(tp), int(fn)],
                [int(fp), int(tn)]]

    acc = (tp + tn) / (tp + tn + fp + fn) * 100
    f1 = (2 * tp) / (2 * tp + fp + fn)
    return acc, f1, c_matrix


def train(device, train_loader, val_loader, model, criterion, optimizer, args):
    best_acc = 0.0
    train_acc_list, val_acc_list, f1_score_list = [], [], []
    best_c_matrix = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        tp, tn, fp, fn = 0, 0, 0, 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            outputs = torch.max(outputs, 1).indices
            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp; tn += sub_tn; fp += sub_fp; fn += sub_fn

        train_acc = (tp + tn) / (tp + tn + fp + fn) * 100
        print(f"Epoch {epoch} - Train Acc: {train_acc:.2f}%")
        train_acc_list.append(train_acc)

        if val_loader is not None:
            val_acc, f1, c_matrix = test(val_loader, model, device)
            val_acc_list.append(val_acc)
            f1_score_list.append(f1)
            print(f"↳ Val Acc: {val_acc:.2f}%, F1: {f1:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_c_matrix = c_matrix
                torch.save(model.state_dict(), f"{args.arch}_best.pth")

    return train_acc_list, val_acc_list, f1_score_list, best_c_matrix


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--dataset", type=str, default="chest_xray")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, batch_size=args.batch_size)
    model = get_model(args.arch, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_acc_list, val_acc_list, f1_score_list, best_c_matrix = train(
        device, train_loader, val_loader, model, criterion, optimizer, args
    )

    print("\n### Final Evaluation on Test Dataset ###")
    test_acc, f1, test_c_matrix = test(test_loader, model, device)
    print(f"Test Acc: {test_acc:.2f}%, F1: {f1:.4f}")

    plot_accuracy(train_acc_list, val_acc_list)
    if len(f1_score_list) > 0:
        plot_f1_score(f1_score_list)
    plot_confusion_matrix(test_c_matrix)
    