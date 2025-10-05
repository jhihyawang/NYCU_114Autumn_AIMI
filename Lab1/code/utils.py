import numpy as np
import matplotlib.pyplot as plt
import torch
import os

import seaborn as sns
import random

def set_seed(seed=39):
    random.seed(seed)                # Python 隨機
    np.random.seed(seed)             # Numpy 隨機
    torch.manual_seed(seed)          # PyTorch CPU
    torch.cuda.manual_seed(seed)     # 單GPU
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

def plot_accuracy(train_acc_list, test_acc_list, model_name):
    plt.figure()
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(test_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    save_path = os.path.join(model_name, "accuracy_curve.png")
    plt.savefig(save_path)
    plt.close()

def plot_f1_score(f1_score_list, model_name):
    plt.figure()
    plt.plot(f1_score_list, label="F1 Score", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("Validation F1-score")
    save_path = os.path.join(model_name, "f1_score_curve.png")
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(confusion_matrix, model_name):
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Normal","Predicted Pneumonia"], yticklabels=["Actual Normal","Actual Pneumonia"])
    plt.title("Confusion Matrix")
    save_path = os.path.join(model_name, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()

def test(test_loader, model, device, criterion=None):
    acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    loss_total = 0.0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if criterion is not None:
                loss_total += criterion(outputs, labels).item()

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

    if criterion is not None:
        avg_loss = loss_total / len(test_loader)
        print(f'↳ Acc.(%): {acc:.2f}%, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}, Avg. Loss: {avg_loss:.6f}')
        return acc, f1_score, c_matrix, avg_loss
    else:
        print(f'↳ Acc.(%): {acc:.2f}%, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        return acc, f1_score, c_matrix