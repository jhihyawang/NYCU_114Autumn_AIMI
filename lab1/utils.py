import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if outputs[j] == 1 and labels[j] == 1:
            tp += 1
        elif outputs[j] == 0 and labels[j] == 0:
            tn += 1
        elif outputs[j] == 1 and labels[j] == 0:
            fp += 1
        elif outputs[j] == 0 and labels[j] == 1:
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(train_acc_list, val_acc_list):
    plt.figure()
    plt.plot(train_acc_list, label="Train Accuracy")
    if len(val_acc_list) > 0:
        plt.plot(val_acc_list, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.show()

def plot_f1_score(f1_score_list):
    plt.figure()
    plt.plot(f1_score_list, label="F1 Score", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Curve")
    plt.legend()
    plt.savefig("f1_curve.png")
    plt.show()

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NORMAL", "PNEUMONIA"],
                yticklabels=["NORMAL", "PNEUMONIA"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()
