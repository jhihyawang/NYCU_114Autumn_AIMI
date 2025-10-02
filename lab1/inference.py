import os
import torch
import torch.nn as nn
from torchvision import models
from argparse import ArgumentParser

from dataloader import get_dataloaders
from utils import test, plot_confusion_matrix


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--dataset", default="chest_xray")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"## Using device: {device} ##")

    # dataloaders
    _, _, test_loader = get_dataloaders(
        data_dir=args.dataset, batch_size=32, resize=224, degree=10
    )

    # model
    model = models.resnet18(pretrained=False) if args.model == "resnet18" else models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(f"{args.model}_best.pt", map_location=device))
    model = model.to(device)

    # test
    acc, f1, c_matrix = test(test_loader, model, device)
    print(f"### Final Test Evaluation ###")
    print(f"Test Acc: {acc:.2f}%, F1: {f1:.4f}")

    # confusion matrix
    plot_confusion_matrix(c_matrix)
