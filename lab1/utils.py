import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

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

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1): tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0): tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0): fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1): fn += 1
    return tp, tn, fp, fn

def plot_accuracy(train_acc_list, val_acc_list):
    plt.figure()
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.savefig("accuracy_curve.png")
    plt.close()

def plot_f1_score(f1_score_list):
    plt.figure()
    plt.plot(f1_score_list, label="Val F1", color="red")
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