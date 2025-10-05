import warnings
from tqdm import tqdm
from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.optim as optim

from models import get_model
from dataloader import get_dataloaders
import torch
from utils import set_seed, measurement, plot_accuracy, plot_f1_score, plot_confusion_matrix, test

def train(device, train_loader, val_loader, model, criterion, optimizer):
    best_loss = float('inf')
    best_model_wts = None
    best_epoch = 0

    train_acc_list = []
    val_acc_list = []
    f1_score_list = []
    best_c_matrix = []

    # Early stopping
    patience = args.patience
    counter = 0

    for epoch in range(1, args.num_epochs + 1):
        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0     
            for _, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

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
            train_acc = (tp + tn) / (tp + tn + fp + fn) * 100
            print(f'Epoch {epoch}/{args.num_epochs}')
            print(f'↳ Loss: {avg_loss:.6f}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')

        print(f'↳ Validation phase') 
        val_acc, f1_score, c_matrix, val_loss = test(val_loader, model, device, criterion) 

        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)
        f1_score_list.append(f1_score)

        # save best model (by lowest val_loss) 
        if val_loss < best_loss: 
            best_loss = val_loss 
            best_epoch = epoch 
            best_model_wts = model.state_dict() 
            best_c_matrix = c_matrix 
            print(f"Best model updated at epoch {epoch} (val_loss: {val_loss:.6f})") 
            counter = 0 
        else: 
            counter += 1 
            print(f"EarlyStopping counter: {counter}/{patience}") 
            if counter >= patience: 
                print("Early stopping triggered. Stop training.") 
                break

    # load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        folder = './weight'
        os.makedirs(folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(folder, f"{args.model}_best.pt"))

    print(f"Training complete. Save at {args.model}_best.pt, Best epoch: {best_epoch}, Best val_loss: {best_loss:.6f}")

    return train_acc_list, val_acc_list, f1_score_list, best_c_matrix

if __name__ == '__main__':
    set_seed(39)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)
    parser.add_argument('--model', type=str, default='densenet121', help='Model name: resnet18, resnet50, vit_base_patch16_224')

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=6, help='Early stopping patience')

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=10)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

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

    # testing
    print("### Final evaluation on test set ###")
    test_acc, f1_score, best_c_matrix = test(test_loader, model, device)

    # plot
    folder = f'./result/{args.model}'
    os.makedirs(folder, exist_ok=True)
    plot_accuracy(train_acc_list, val_acc_list, folder)
    plot_f1_score(f1_score_list, folder)
    plot_confusion_matrix(best_c_matrix, folder)