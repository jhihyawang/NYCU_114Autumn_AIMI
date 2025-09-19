import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

def calculate_accuracy(y_true, y_pred):
    return (y_true == y_pred).sum().item() / len(y_true)

def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")

def plot_metrics(train_acc, train_f1, val_acc=None, val_f1=None):
    epochs = range(1, len(train_acc) + 1)

    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    if val_acc: plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.legend()
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.title("Accuracy Trend")
    plt.show()

    plt.figure()
    plt.plot(epochs, train_f1, label="Train F1")
    if val_f1: plt.plot(epochs, val_f1, label="Val F1")
    plt.legend()
    plt.xlabel("Epochs"); plt.ylabel("F1-score"); plt.title("F1 Trend")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
