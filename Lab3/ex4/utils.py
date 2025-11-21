import numpy as np
import matplotlib.pyplot as plt
import torch
import os

import seaborn as sns
import random

def set_seed(seed=39):
    random.seed(seed)
    np.random.seed(seed)             
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
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

def plot_trainning_loss(train_loss_list, model_name):
    plt.figure()
    plt.plot(train_loss_list, label="Train Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    save_path = os.path.join(model_name, "training_loss_curve.png")
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

def plot_confusion_matrix(confusion_matrix, save_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "bacteria", "virus", "COVID-19"],
        yticklabels=["normal", "bacteria", "virus", "COVID-19"]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    # 儲存圖檔
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")