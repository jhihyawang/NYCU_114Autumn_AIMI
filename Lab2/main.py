import copy
from xml.parsers.expat import model
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import EEGNet 
from models.DeepConvNet import DeepConvNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
import random
from dataloader import BCIDataset
from torch.utils.data import DataLoader

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_train_acc(train_acc_list, epochs, saving_dir='results/'):
    # plot training accuracy
    train_acc_list = np.array(train_acc_list)
    plt.figure()
    plt.plot(np.arange(1, epochs+1), train_acc_list, label='Train Acc.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    plt.savefig(f'{saving_dir}/train_acc.png')
    plt.close()


def plot_train_loss(train_loss_list, epochs, saving_dir='results/'):
    train_loss_list = np.array(train_loss_list)
    plt.figure()
    plt.plot(np.arange(1, epochs+1), train_loss_list, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(f'{saving_dir}/train_loss.png')
    plt.close()

def plot_test_acc(test_acc_list, epochs, saving_dir='results/'):
    # plot testing accuracy
    test_acc_list = np.array(test_acc_list)
    plt.figure()
    plt.plot(np.arange(1, epochs+1), test_acc_list, label='Test Acc.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Testing Accuracy')
    plt.legend()
    plt.savefig(f'{saving_dir}/test_acc.png')
    plt.close()

def train(model, loader, test_loader, criterion, optimizer, args, device):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    for epoch in range(1, args.num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        test_acc = test(model, test_loader, device)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = model.state_dict()
        print(f'Test Acc. (%): {test_acc:3.2f}%')

    torch.save(best_wts, f'{args.saving_dir}/best.pt')
    return avg_acc_list, avg_loss_list, test_acc_list


def test(model, loader, device):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=150)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-optimizer", type=str, default="Adam")
    parser.add_argument("-weight_decay", type=float, default=1e-4)
    parser.add_argument("-dropout", type=float, default=0.5)
    parser.add_argument("-model", type=str, default="EEGNet")
    parser.add_argument("-experiment_id", type=str, default="exp_001")
    args = parser.parse_args()
 
    set_seed(456)

    # Create experiment directory
    import os
    args.saving_dir = f'results/{args.experiment_id}'
    os.makedirs(args.saving_dir, exist_ok=True)
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Saving results to: {args.saving_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model.lower() == "deepconvnet":
        model = DeepConvNet(dropout_rate=args.dropout)
    else:      
        model = EEGNet(dropout_rate=args.dropout)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.to(device)
    criterion.to(device)

    train_acc_list, train_loss_list, test_acc_list = train(model, train_loader, test_loader, criterion, optimizer, args, device)

    plot_train_acc(train_acc_list, args.num_epochs, args.saving_dir)
    plot_train_loss(train_loss_list, args.num_epochs, args.saving_dir)
    plot_test_acc(test_acc_list, args.num_epochs, args.saving_dir)
    
    # Save experiment configuration
    config_file = os.path.join(args.saving_dir, 'config.txt')
    with open(config_file, 'w') as f:
        f.write(f"Experiment ID: {args.experiment_id}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.num_epochs}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Final Test Accuracy: {test_acc_list[-1]:.2f}%\n")
        f.write(f"Best Test Accuracy: {max(test_acc_list):.2f}%\n")
    
    print(f"Experiment {args.experiment_id} completed!")
    print(f"Final test accuracy: {test_acc_list[-1]:.2f}%")
    print(f"Best test accuracy: {max(test_acc_list):.2f}%")
    print(f"Results saved to: {args.saving_dir}")