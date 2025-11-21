import warnings
from tqdm import tqdm
from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import cv2

from model import build_model
from dataloader import (
    get_train_val_loaders,
    get_test_loader,
    build_transforms,
    CLASS_NAMES
)

from utils import (
    set_seed, 
    measurement, 
    plot_accuracy, 
    plot_f1_score, 
    plot_confusion_matrix, 
    plot_trainning_loss
)

from loss import get_criterion

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# Optimizer Selection
# ============================================================
def get_optimizer(args, model):
    """Get optimizer based on args"""
    if args.optimizer == 'adam':
        print(f"🔧 Using Adam optimizer (lr={args.lr}, weight_decay={args.wd})")
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adamw':
        print(f"🔧 Using AdamW optimizer (lr={args.lr}, weight_decay={args.wd})")
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        print(f"🔧 Using SGD optimizer (lr={args.lr}, momentum=0.9, weight_decay={args.wd})")
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'rmsprop':
        print(f"🔧 Using RMSprop optimizer (lr={args.lr}, weight_decay={args.wd})")
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

# ============================================================
# Learning Rate Scheduler
# ============================================================
def get_scheduler(args, optimizer):
    """Get learning rate scheduler based on args"""
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-6
        )
        print(f"📈 Using CosineAnnealingLR (T_max={args.num_epochs}, eta_min=1e-6)")
        
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
        print(f"📈 Using StepLR (step_size={args.step_size}, gamma={args.gamma})")
        
    elif args.scheduler == "reduce":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=args.gamma,
            patience=args.scheduler_patience,
            min_lr=1e-7
        )
        print(f"📈 Using ReduceLROnPlateau (mode=max, factor={args.gamma}, patience={args.scheduler_patience})")
    elif args.scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        print(f"📈 Using ExponentialLR (gamma={args.gamma})")
        
    elif args.scheduler == "multistep":
        milestones = [int(args.num_epochs * 0.5), int(args.num_epochs * 0.75)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
        print(f"📈 Using MultiStepLR (milestones={milestones}, gamma={args.gamma})")
    else:
        scheduler = None
        print("📈 No scheduler used")
        
    return scheduler

# ============================================================
# Validation Function
# ============================================================
def validate(val_loader, model, device, criterion, pred_weights=None):
    """
    Validate model and return accuracy, macro-F1, confusion matrix, loss, predictions, and labels
    
    Args:
        pred_weights: Optional tensor of shape (num_classes,) to weight predictions
                      Lower weights reduce the chance of that class being predicted
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            # Apply prediction weights if provided
            if pred_weights is not None:
                softmax_output = torch.softmax(outputs, dim=1)
                weighted_output = softmax_output * pred_weights.to(device)
                preds = torch.argmax(weighted_output, dim=1)
            else:
                preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Accuracy
    correct = (all_preds == all_labels).sum()
    val_acc = correct / len(all_labels) * 100
    
    # Macro F1-score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # Confusion matrix
    c_matrix = confusion_matrix(all_labels, all_preds)
    
    return val_acc, f1_macro, c_matrix, val_loss, all_preds, all_labels

# ============================================================
# Training Function
# ============================================================
def train(device, train_loader, val_loader, model, criterion, optimizer, scheduler, args, exp_dir):
    """
    Main training loop with training loss as primary metric
    """
    # best_f1 = 0.0
    best_f1 = 0.0  # Use training loss instead
    best_model_wts = None
    best_epoch = 0
    best_c_matrix = None
    
    train_acc_list, train_loss_list, val_acc_list, f1_score_list = [], [], [], []
    
    patience_counter = 0
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(exp_dir / "tensorboard"))
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()

        epoch_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * inputs.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        epoch_loss /= len(train_loader.dataset)
        train_acc = (np.array(all_train_preds) == np.array(all_train_labels)).sum() / len(all_train_labels) * 100
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*80}")
        print(f"📊 Training - Loss: {epoch_loss:.6f} | Acc: {train_acc:.2f}% | Macro-F1: {train_f1:.4f}")
        
        # Validation
        print("🔍 Validating...")
        val_acc, f1_macro, c_matrix, val_loss, val_preds, val_labels = validate(
            val_loader, model, device, criterion, pred_weights=args.pred_weights
        )
        
        print(f"📊 Validation - Loss: {val_loss:.6f} | Acc: {val_acc:.2f}% | Macro-F1: {f1_macro:.4f}")
        
        # Classification report for validation
        val_report = classification_report(
            val_labels, val_preds, target_names=CLASS_NAMES, digits=4
        )
        print(f"\n📋 Validation Classification Report (Epoch {epoch}):")
        print(val_report)
        print(f"{'='*80}\n")
        
        # Store metrics
        train_acc_list.append(train_acc)
        train_loss_list.append(epoch_loss)
        val_acc_list.append(val_acc)
        f1_score_list.append(f1_macro)
        
        # TensorBoard logging
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("F1/val", f1_macro, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        
        # Per-class F1 scores
        val_report_dict = classification_report(
            val_labels, val_preds, target_names=CLASS_NAMES, output_dict=True, digits=4
        )
        for cls in CLASS_NAMES:
            writer.add_scalar(f"PerClassF1/{cls}", val_report_dict[cls]["f1-score"], epoch)
        
        # Learning rate scheduler step
        if scheduler is not None:
            if args.scheduler == "reduce":
                scheduler.step(f1_macro)
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"📉 Current Learning Rate: {current_lr:.2e}")
        
        # Save best model based on training loss
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_epoch = epoch
            best_model_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_c_matrix = c_matrix
            patience_counter = 0
            print(f"✅ Best model updated! (Epoch {epoch}, Val Macro-F1: {f1_macro:.4f})")
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     best_epoch = epoch
        #     best_model_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        #     best_c_matrix = c_matrix
        #     patience_counter = 0
        #     print(f"✅ Best model updated! (Epoch {epoch}, Train Loss: {epoch_loss:.6f})")
        else:
            patience_counter += 1
            print(f"⏳ Early stopping counter: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch}")
                break
    
    # Close TensorBoard writer
    writer.close()
    
    # Save best model
    best_model_path = None
    if best_model_wts is not None:
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = weights_dir / "best.pt"
        
        # Load best weights back to model
        model.load_state_dict(best_model_wts, strict=True)
        torch.save(model.state_dict(), best_model_path)
        print(f"\n💾 Saved best model to {best_model_path}")
        print(f"   Best Epoch: {best_epoch} | Best Val Macro-F1: {best_f1:.4f}")
        # print(f"   Best Epoch: {best_epoch} | Best Train Loss: {best_loss:.6f}")
        
        # Generate final classification report for best model
        print("\n" + "="*80)
        print("📊 Final Validation Results (Best Model)")
        print("="*80)
        _, _, _, _, best_preds, best_labels = validate(val_loader, model, device, criterion, pred_weights=args.pred_weights)
        best_report = classification_report(
            best_labels, best_preds, target_names=CLASS_NAMES, digits=4
        )
        print(best_report)
        
        # Save classification report to report.txt
        report_path = "report.txt"
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"Experiment ID: {args.experiment_id}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Loss: {args.loss_type}\n")
            f.write(f"Optimizer: {args.optimizer}\n")
            f.write(f"Scheduler: {args.scheduler}\n")
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Val Macro-F1: {best_f1:.4f}\n")
            # f.write(f"Best Train Loss: {best_loss:.6f}\n")
            f.write(f"\n{'='*80}\n")
            f.write("Classification Report for Best Model\n")
            f.write(f"{'='*80}\n\n")
            f.write(best_report)
        
        print(f"📝 Classification report saved to {report_path}")
    
    return train_acc_list, train_loss_list, val_acc_list, f1_score_list, best_c_matrix, best_model_path

# ============================================================
# Test Inference Function
# ============================================================
def inference(model, device, dataset_path, img_size, output_dir, experiment_id, pred_weights=None):
    """
    Perform inference on test set and save results to CSV
    
    Args:
        pred_weights: Optional tensor of shape (num_classes,) to weight predictions
    """
    print("\n" + "="*80)
    print("🚀 Starting Test Inference")
    print("="*80)
    
    if pred_weights is not None:
        print(f"📊 Using prediction weights: {pred_weights.cpu().numpy()}")
    
    transform = build_transforms(img_size)
    
    test_csv = "../csv/test_data_sample.csv"
    img_dir = Path(dataset_path) / "test_images"
    
    df_test = pd.read_csv(test_csv)
    if "new_filename" not in df_test.columns:
        raise ValueError("❌ test_data_sample.csv must contain 'new_filename' column")
    
    model.eval()
    results = []
    
    for fname in tqdm(df_test["new_filename"], desc="🔍 Predicting"):
        img_path = img_dir / fname
        
        if not img_path.exists():
            print(f"⚠️ Image not found: {img_path}")
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ Cannot read image: {img_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=image)
        image_tensor = transformed["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            
            # Apply prediction weights if provided
            if pred_weights is not None:
                softmax_output = torch.softmax(output, dim=1)
                weighted_output = softmax_output * pred_weights.to(device)
                pred = torch.argmax(weighted_output, dim=1).item()
            else:
                pred = torch.argmax(output, dim=1).item()
        
        # Create row for CSV
        row = {"new_filename": fname}
        for i, cls in enumerate(CLASS_NAMES):
            row[cls] = 1 if i == pred else 0
        results.append(row)
    
    # Save results to CSV
    output_csv = output_dir / f"test_predictions_{experiment_id}.csv"
    df_out = pd.DataFrame(results, columns=["new_filename"] + CLASS_NAMES)
    df_out.to_csv(output_csv, index=False)
    
    print(f"✅ Test inference complete!")
    print(f"📄 Results saved to: {output_csv}")
    print(f"   Total predictions: {len(results)}")
    
    # Print prediction distribution
    pred_dist = df_out[CLASS_NAMES].sum()
    print(f"\n📊 Prediction Distribution:")
    for cls in CLASS_NAMES:
        print(f"   {cls}: {pred_dist[cls]}")
    print("="*80 + "\n")

# ============================================================
# Main Function
# ============================================================
if __name__ == '__main__':
    set_seed(42)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    parser = ArgumentParser()
    
    # Basic settings
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                        help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-3,
                        help='Weight decay')
    
    # Loss function settings
    parser.add_argument('--loss_type', type=str, default='ce',
    choices=['ce', 'wce', 'focal', 'label_smooth', 'weighted_label_smooth'], help='Loss function type')

    parser.add_argument('--use_class_weight', action='store_true',
                        help='Use class weights in loss function')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss')
    parser.add_argument('--label_smooth_eps', type=float, default=0.2,
                        help='Epsilon for Label Smoothing')
    
    # Confusion-Aware Loss settings
    parser.add_argument('--confusion_update_freq', type=int, default=5,
                        help='Update frequency for confusion weights (epochs)')
    parser.add_argument('--confusion_penalty', type=float, default=2.0,
                        help='Penalty coefficient for confused class pairs')
    parser.add_argument('--confusion_smoothing', type=float, default=0.1,
                        help='Smoothing factor for confusion weights')
    
    # Scheduler settings
    parser.add_argument('--scheduler', type=str, default=None, help='Learning rate scheduler type')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for learning rate decay')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau scheduler')
    
    # Training settings
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='../no_exp/ori',
                        help='Dataset root directory')
    
    # Experiment settings
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='Experiment ID (folder name). If None, auto-generated.')
    
    # Prediction weights settings
    parser.add_argument('--pred_weights', type=str, default=None,
                        help='Comma-separated prediction weights for each class (e.g., "1.0,1.0,0.7,1.0"). Lower values reduce prediction probability.')
    parser.add_argument('--train_csv', type=str, default="../csv/train_data.csv",
                        help='Path to the training CSV file')
    args = parser.parse_args()
    
    # Parse prediction weights if provided
    if args.pred_weights is not None:
        try:
            weights_list = [float(w.strip()) for w in args.pred_weights.split(',')]
            if len(weights_list) != args.num_classes:
                raise ValueError(f"Number of weights ({len(weights_list)}) must match num_classes ({args.num_classes})")
            args.pred_weights = torch.tensor(weights_list)
            print(f"📊 Prediction weights: {args.pred_weights.numpy()}")
        except Exception as e:
            raise ValueError(f"Error parsing pred_weights: {e}")
    else:
        args.pred_weights = None
    
    # Generate experiment ID if not provided
    if args.experiment_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_id = f"{args.model}_lr{args.lr}_bs{args.batch_size}_{args.loss_type}_{args.optimizer}_sch{args.scheduler}_{timestamp}"
    
    # Create experiment directory
    exp_dir = Path("result") / args.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🚀 Starting Experiment: {args.experiment_id}")
    print("="*80)
    print(f"📁 Experiment directory: {exp_dir.resolve()}")
    print(f"🏗️  Model: {args.model}")
    print(f"⚖️  Loss: {args.loss_type}")
    print(f"🔧 Optimizer: {args.optimizer} (lr={args.lr}, wd={args.wd})")
    print(f"📈 Scheduler: {args.scheduler}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.num_epochs}")
    print(f"⚡ AMP: {args.use_amp}")
    if args.pred_weights is not None:
        print(f"📊 Prediction weights: {args.pred_weights.numpy()}")
    print("="*80 + "\n")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.backends.cudnn.benchmark = True
    
    # Load data
    print("\n📂 Loading datasets...")
    train_loader, val_loader, class_weights = get_train_val_loaders(
        data_root=args.dataset,
        img_size=args.resize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_csv=args.train_csv
    )
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    # print(f"   Class weights: {class_weights.numpy()}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = build_model(model_name=args.model, pretrained=True, num_classes=args.num_classes)
    model = model.to(device)

    # Get loss function
    criterion = get_criterion(args, device, class_weights)
    
    # Get optimizer
    optimizer = get_optimizer(args, model)
    
    # Get scheduler
    scheduler = get_scheduler(args, optimizer)
    
    # Train model
    print("\n🎯 Starting training...")
    train_acc_list, train_loss_list, val_acc_list, f1_score_list, best_c_matrix, best_model_path = train(
        device, train_loader, val_loader, model, criterion, optimizer, scheduler, args, exp_dir
    )
    
    # Generate and save plots
    print("\n📊 Generating plots...")
    plot_accuracy(train_acc_list, val_acc_list, exp_dir)
    plot_trainning_loss(train_loss_list, exp_dir)
    plot_f1_score(f1_score_list, exp_dir)
    plot_confusion_matrix(best_c_matrix, exp_dir)
    print("✅ Plots saved!")
    
    # Perform inference on test set
    inference(model, device, args.dataset, args.resize, exp_dir, args.experiment_id, pred_weights=args.pred_weights)
    
    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print("="*80)
    print(f"📁 All results saved in: {exp_dir.resolve()}")
    print(f"   - Best model weights: {best_model_path}")
    print(f"   - Classification report: {exp_dir / 'report.txt'}")
    print(f"   - Confusion matrix: {exp_dir / 'confusion_matrix.png'}")
    print(f"   - Training plots: accuracy_curve.png, f1_score_curve.png, training_loss_curve.png")
    print(f"   - Test predictions: test_predictions_{args.experiment_id}.csv")
    print(f"   - TensorBoard logs: {exp_dir / 'tensorboard'}")
    print("="*80 + "\n")
