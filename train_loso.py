import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
import torch.nn as nn
import yaml
import argparse
import time
from scipy.signal import butter, filtfilt
from sklearn.metrics import cohen_kappa_score, f1_score
import copy
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils.util import set_seed, setup_logger, classwise_augmentation
from utils.load_data import load_data
from utils.load_model import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    return parser.parse_args()

def train_test_loso(data, labels, config, device, logger):
    """
    LOSO Training & Testing
    Args:
        data: numpy array, shape (n_subjects, n_epochs, n_channels, n_times)
        labels: numpy array, shape (n_subjects, n_epochs)
        config: configuration dictionary
        device: torch device
        logger: logger object
    """
    n_subjects = len(data)
    batch_size = config['train']['batch_size']
    num_epochs = config['train']['epochs']

    logger.info("========== Start LOSO Training ==========")
    
    all_val_subject_acc = []
    all_val_subject_kappa = []
    all_val_subject_f1 = []
    all_test_subject_acc = []
    all_test_subject_kappa = []
    all_test_subject_f1 = []
    for test_subj in range(n_subjects):
        # Model initialization
        model_params = config['model']

        # train
        train_data_list = [data[i] for i in range(n_subjects) if i != test_subj]
        train_labels_list = [labels[i] for i in range(n_subjects) if i != test_subj]

        # valid
        val_data_list = []
        val_labels_list = []
        new_train_data_list = []
        new_train_labels_list = []
        for data_subj, labels_subj in zip(train_data_list, train_labels_list):
            X_train, X_val, y_train, y_val = train_test_split(
                data_subj,
                labels_subj,
                test_size=0.2,
                stratify=labels_subj,
                random_state=config['train']['seed']
            )
            new_train_data_list.append(X_train)
            new_train_labels_list.append(y_train)
            val_data_list.append(X_val)
            val_labels_list.append(y_val)

        train_data = np.concatenate(new_train_data_list, axis=0)
        train_labels = np.concatenate(new_train_labels_list, axis=0)
        valid_data = np.concatenate(val_data_list, axis=0)
        valid_labels = np.concatenate(val_labels_list, axis=0)

        # test
        test_data = data[test_subj]
        test_labels = labels[test_subj]

        # flatten subjects into epochs
        train_data = train_data[:, None, :, :]
        valid_data = valid_data[:, None, :, :]
        test_data  = test_data[:, None, :, :]
        
        if config['train']['norm'] == 'Z_Score':
            # Z-score normalization
            train_mean = train_data.mean(axis=(0, 1, 3), keepdims=True) 
            train_std  = train_data.std(axis=(0, 1, 3), keepdims=True)
            train_data = (train_data - train_mean) / train_std
            valid_data = (valid_data - train_mean) / train_std
            test_data  = (test_data - train_mean) / train_std
        
        # model
        model, criterion, optimizer = load_model(model_params, config, n_class, n_channels, n_times, logger, device)
        
        logger.info(f"====== Subject {test_subj + 1}/{n_subjects} as Test ======")

        param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model Parameters: {param}")
        
        logger.info(f"Train size = {train_data.shape[0]}, Valid size = {valid_data.shape[0]}, Test size = {test_data.shape[0]}")

        # Create Dataset
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long)
        )

        valid_dataset = torch.utils.data.TensorDataset(
            torch.tensor(valid_data, dtype=torch.float32),
            torch.tensor(valid_labels, dtype=torch.long)
        )

        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_data, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True, persistent_workers=True)

        # Training
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_test_acc = 0.0
        best_test_loss = float('inf')
        best_val_model_state = None
        best_test_model_state = None
        best_val_epoch = 0
        best_test_epoch = 0
        patient = 0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for X, y in train_loader:
                # Data augmentation
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                X, y = classwise_augmentation(X, y, n_segments=8)

                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
            
            train_acc = correct / total
            train_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_val, y_val in valid_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, dim=1)
                    val_correct += (preds == y_val).sum().item()
                    val_total += y_val.size(0)

            val_acc = val_correct / val_total
            val_loss /= len(valid_loader)

            if val_acc > best_val_acc or val_acc == best_val_acc and val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_model_state = copy.deepcopy(model.state_dict())
                best_val_epoch = epoch + 1
                patient = 0
                logger.info(f"Sub {test_subj+1} | Early Stopping: Best Epoch {epoch+1}/{num_epochs}")
            else:
                patient +=1    
            
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for X_test, y_test in test_loader:
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    outputs = model(X_test)
                    loss = criterion(outputs, y_test)
                    test_loss += loss.item()

                    _, preds = torch.max(outputs, dim=1)
                    test_correct += (preds == y_test).sum().item()
                    test_total += y_test.size(0)

            test_acc = test_correct / test_total
            test_loss /= len(test_loader)

            if test_acc > best_test_acc or test_acc == best_test_acc and test_loss < best_test_loss:
                best_test_acc = test_acc
                best_test_loss = test_loss
                best_test_epoch = epoch + 1
                best_test_model_state = copy.deepcopy(model.state_dict())
                logger.info(f"Sub {test_subj+1} | All Epoch: Best Epoch {epoch+1}/{num_epochs}")

            logger.info(f"Sub {test_subj+1} | Epoch {epoch+1}/{num_epochs} | train Loss: {train_loss:.4f} | train Accuracy: {train_acc:.4f} | valid Loss: {val_loss:.4f} | valid Accuracy: {val_acc:.4f} | test Accuracy: {test_acc:.4f} | patient: {patient}")

        # Early Stopping Test
        save_dir = f"/mnt/data2/DSAINet/weight/{config['model']['name']}/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_val_model_state, f"/mnt/data2/DSAINet/weight/{config['model']['name']}/{args.dataset}/{test_subj}.pth")
        
        model.load_state_dict(best_val_model_state)
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.cpu().numpy().tolist())
        # acc
        acc = correct / total
        all_val_subject_acc.append(acc)
        # kappa
        kappa = cohen_kappa_score(all_targets, all_preds)
        all_val_subject_kappa.append(kappa)
        # f1
        f1 = f1_score(all_targets, all_preds, average='weighted')
        all_val_subject_f1.append(f1)
        logger.info(f"Early Stopping Best Epoch: {best_val_epoch} | Test Subject {test_subj + 1} | Accuracy: {acc:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

        # All Epochs Test
        save_dir = f"/mnt/data2/DSAINet/weight/{config['model']['name']}/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_test_model_state, f"/mnt/data2/DSAINet/weight/{config['model']['name']}/{args.dataset}/best_{test_subj}.pth")
        model.load_state_dict(best_test_model_state)
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.cpu().numpy().tolist())
        # acc
        acc = correct / total
        all_test_subject_acc.append(acc)
        # kappa
        kappa = cohen_kappa_score(all_targets, all_preds)
        all_test_subject_kappa.append(kappa)
        # f1
        f1 = f1_score(all_targets, all_preds, average='weighted')
        all_test_subject_f1.append(f1)
        logger.info(f"All Best Epoch: {best_test_epoch} | Test Subject {test_subj + 1} | Accuracy: {acc:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

    logger.info("========== LOSO Done ==========")

    logger.info("----- Early Stopping Results -----")
    for i, (acc, kappa, f1) in enumerate(zip(all_val_subject_acc, all_val_subject_kappa, all_val_subject_f1)):
        logger.info(f"Subject {i+1:02d} | Acc = {acc:.4f} | Kappa = {kappa:.4f} | F1 = {f1:.4f}")

    logger.info("----- All Epoch Results -----")
    for i, (acc, kappa, f1) in enumerate(zip(all_test_subject_acc, all_test_subject_kappa, all_test_subject_f1)):
        logger.info(f"Subject {i+1:02d} | Acc = {acc:.4f} | Kappa = {kappa:.4f} | F1 = {f1:.4f}")

    logger.info(f"Early Stopping - Average LOSO Accuracy: {np.mean(all_val_subject_acc):.4f}±{np.std(all_val_subject_acc):.4f}")
    logger.info(f"Early Stopping - Average LOSO Kappa: {np.mean(all_val_subject_kappa):.4f}±{np.std(all_val_subject_kappa):.4f}")
    logger.info(f"Early Stopping - Average LOSO F1: {np.mean(all_val_subject_f1):.4f}±{np.std(all_val_subject_f1):.4f}")

    logger.info(f"All Epoch - Average LOSO Accuracy: {np.mean(all_test_subject_acc):.4f}±{np.std(all_test_subject_acc):.4f}")
    logger.info(f"All Epoch - Average LOSO Kappa: {np.mean(all_test_subject_kappa):.4f}±{np.std(all_test_subject_kappa):.4f}")
    logger.info(f"All Epoch - Average LOSO F1: {np.mean(all_test_subject_f1):.4f}±{np.std(all_test_subject_f1):.4f}")

if __name__ == "__main__":
    args = parse_args()
    # config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # seed
    if args.seed is not None:
        set_seed(args.seed)
        config['train']['seed'] = args.seed
    else:
        set_seed(config['train']['seed'])
    
    # batch size
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size

    # epoch
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    
    # learning rate
    if args.lr is not None:
        config['train']['lr'] = args.lr
    
    # sample rate
    if args.dataset in ['BCIC_IV_2a', 'BCIC_IV_2b', 'OpenBMI', 'PhysioNet_MI']:
        sample_rate = 250
    elif args.dataset in ['Zhou2016']:
        sample_rate = 200

    # logger
    logger = setup_logger(f"{args.dataset}", log_dir=f"./log/{config['model']['name']}/", overwrite=True)
    # device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # log config 
    logger.info("==== Experiment Config ====")
    logger.info(yaml.dump(config).rstrip())
    logger.info("==========================")

    # Load data
    data, labels, n_class, n_channels, n_times = load_data(args.dataset, 'LOSO')

    train_test_loso(data, labels, config, device, logger)