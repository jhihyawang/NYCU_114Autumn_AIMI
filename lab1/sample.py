import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator

from models import get_model
from dataloader import get_dataloaders
import torch
import numpy as np
import random

def set_seed(seed=39):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(train_acc_list, test_acc_list):
    plt.figure()
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(test_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.savefig("accuracy_curve.png")
    plt.close()

def plot_f1_score(f1_score_list):
    plt.figure()
    plt.plot(f1_score_list, label="F1 Score", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("Validation F1-score")
    plt.savefig("f1_score_curve.png")
    plt.close()

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Pneumonia"], yticklabels=["Normal","Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

def train(device, train_loader, val_loader, model, criterion, optimizer):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    f1_score_list = []
    best_c_matrix = []

    # Earlystopping
    patience = 10
    counter = 0

    for epoch in range(1, args.num_epochs+1):

        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0     
            for _, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn          

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')

        print (f'↳ Validation phase')
        val_acc, f1_score, c_matrix = test(val_loader, model, device)
        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)
        f1_score_list.append(f1_score)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            best_c_matrix = c_matrix
            torch.save(best_model_wts, f"{args.model}_best.pt")
            print(f"Best model updated with val_acc: {val_acc:.2f}%")
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered, Stop training.")
                break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "model_weights.pt")

    return train_acc_list, val_acc_list, f1_score_list, best_c_matrix

def test(test_loader, model, device):
    acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tn), int(fp)],
                    [int(fn), int(tp)]]
        acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Acc.(%): {acc:.2f}%, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')

    return acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)
    parser.add_argument('--model', type=str, default='densenet121', help='Model name: resnet18, resnet50, densenet121, efficientnet_b0')

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=10)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')
    set_seed(42)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.dataset,
        batch_size=args.batch_size,
        resize=args.resize,
        degree=args.degree
    )
    model = get_model(args.model, args.num_classes)

    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training
    train_acc_list, val_acc_list, f1_score_list, best_c_matrix = train(device, train_loader, val_loader, model, criterion, optimizer)
    #val_acc, f1_score, c_matrix = test(val_loader, model, device)
    # testing
    print("### Final evaluation on test set ###")
    test_acc, f1_score, best_c_matrix = test(test_loader, model, device)

    # plot
    plot_accuracy(train_acc_list, val_acc_list)
    plot_f1_score(f1_score_list)
    plot_confusion_matrix(best_c_matrix)