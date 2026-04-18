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
import random
import time
from sklearn.metrics import cohen_kappa_score, f1_score
import copy
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
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

def train_test_kfold(data, labels, config, device, logger):
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

    fold_num = 0
    if args.dataset == 'Mumtaz2016':
        fold_num = 10
        subjects_idx = np.arange(n_subjects)
        hc_idx = subjects_idx[:30]
        mdd_idx = subjects_idx[30:] 
        np.random.shuffle(hc_idx)
        np.random.shuffle(mdd_idx)
        hc_folds = np.array_split(hc_idx, fold_num)
        mdd_folds = np.array_split(mdd_idx, fold_num)
        folds = []
        for i in range(fold_num):
            fold = np.concatenate((hc_folds[i], mdd_folds[i]))
            folds.append(fold)
    elif args.dataset == 'ADFTD':
        fold_num = 10
        subjects_idx = np.arange(n_subjects)
        ad_idx = subjects_idx[:36]
        cn_idx = subjects_idx[36:65]
        ftd_idx = subjects_idx[65:]
        np.random.shuffle(ad_idx)
        np.random.shuffle(cn_idx)
        np.random.shuffle(ftd_idx)
        ad_folds = np.array_split(ad_idx, fold_num)
        cn_folds = np.array_split(cn_idx, fold_num)
        ftd_folds = np.array_split(ftd_idx, fold_num)
        folds = []
        for i in range(fold_num):
            fold = np.concatenate((ad_folds[i], cn_folds[i], ftd_folds[i]))
            folds.append(fold)
    elif args.dataset == 'Rockhill2021':
        fold_num = 5
        subjects_idx = np.arange(n_subjects)
        hc_idx = [0, 1, 3, 6, 7, 9, 16, 18, 19, 22, 23, 26, 27, 28, 29, 30]
        pd_idx = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15, 17, 20, 21, 24, 25]
        np.random.shuffle(hc_idx)
        np.random.shuffle(pd_idx)
        hc_folds = np.array_split(hc_idx, fold_num)
        pd_folds = np.array_split(pd_idx, fold_num)
        folds = []
        for i in range(fold_num):
            fold = np.concatenate((hc_folds[i], pd_folds[i]))
            folds.append(fold)
    else:
        fold_num = 10
        subjects_idx = np.arange(n_subjects)
        np.random.shuffle(subjects_idx)
        folds = np.array_split(subjects_idx, 10)

    logger.info("========== Folds Split Result ==========")
    for i, fold in enumerate(folds):
        fold_ids = np.array(fold, dtype=int).tolist()
        logger.info(f"[FOLDS] Fold {i:02d}: {fold_ids}")

    logger.info(f"========== Start {fold_num}-Fold Training ==========")

    all_val_fold_acc = []
    all_val_fold_kappa = []
    all_val_fold_f1 = []
    all_test_fold_acc = []
    all_test_fold_kappa = []
    all_test_fold_f1 = []
    for test_fold_id in range(fold_num):
        logger.info(f"====== Fold {test_fold_id + 1}/{fold_num} as Test ======")

        # Model initialization
        model_params = config['model']

        # split id
        test_idx = folds[test_fold_id]
        other_folds = [folds[i] for i in range(fold_num) if i != test_fold_id]
        val_fold = random.choice(other_folds)      
        val_idx = np.array(val_fold)
        train_folds = [f for f in other_folds if not np.array_equal(f, val_fold)]
        train_idx = np.concatenate(train_folds)

        logger.info(f"Test subjects:  {test_idx.tolist()}")
        logger.info(f"Val subjects:   {val_idx.tolist()}")
        logger.info(f"Train subjects: {train_idx.tolist()}")

        # train
        train_data_list = [data[i] for i in train_idx]
        train_labels_list = [labels[i] for i in train_idx]
        # valid
        valid_data_list = [data[i] for i in val_idx]
        valid_labels_list = [labels[i] for i in val_idx]
        # test
        test_data_list = [data[i] for i in test_idx]
        test_labels_list = [labels[i] for i in test_idx]

        train_data = np.concatenate(train_data_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        valid_data = np.concatenate(valid_data_list, axis=0)
        valid_labels = np.concatenate(valid_labels_list, axis=0)
        test_data = np.concatenate(test_data_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)

        # flatten subjects into epochs
        train_data = train_data.reshape(-1, 1, n_channels, n_times)
        valid_data = valid_data.reshape(-1, 1, n_channels, n_times)
        test_data = test_data.reshape(-1, 1, n_channels, n_times)
        
        if config['train']['norm'] == 'Z_Score':
            # Z-score normalization
            train_mean = train_data.mean(axis=(0, 1, 3), keepdims=True) 
            train_std  = train_data.std(axis=(0, 1, 3), keepdims=True)
            train_data = (train_data - train_mean) / train_std
            valid_data = (valid_data - train_mean) / train_std
            test_data  = (test_data - train_mean) / train_std

        # model
        model, criterion, optimizer = load_model(model_params, config, n_class, n_channels, n_times, logger, device)
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

            # ADFTD: to confirm validation stable
            if args.dataset == 'ADFTD' and epoch >= 10:
                if val_acc > best_val_acc or val_acc == best_val_acc and val_loss < best_val_loss:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_val_model_state = copy.deepcopy(model.state_dict())
                    best_val_epoch = epoch + 1
                    patient = 0
                    logger.info(f"Fold {test_fold_id+1} | Early Stopping: Best Epoch {epoch+1}/{num_epochs}")
                else:
                    patient +=1   
            # Rockhill2021: to confirm validation stable
            elif args.dataset == 'Rockhill2021' and epoch >= 15:
                if val_acc > best_val_acc or val_acc == best_val_acc and val_loss < best_val_loss:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_val_model_state = copy.deepcopy(model.state_dict())
                    best_val_epoch = epoch + 1
                    patient = 0
                    logger.info(f"Fold {test_fold_id+1} | Early Stopping: Best Epoch {epoch+1}/{num_epochs}")
                else:
                    patient +=1   
            elif args.dataset != 'ADFTD' and args.dataset != 'Rockhill2021':
                if val_acc > best_val_acc or val_acc == best_val_acc and val_loss < best_val_loss:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_val_model_state = copy.deepcopy(model.state_dict())
                    best_val_epoch = epoch + 1
                    patient = 0
                    logger.info(f"Fold {test_fold_id+1} | Early Stopping: Best Epoch {epoch+1}/{num_epochs}")
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
                logger.info(f"Fold {test_fold_id+1} | All Epoch: Best Epoch {epoch+1}/{num_epochs}")

            logger.info(f"Fold {test_fold_id+1} | Epoch {epoch+1}/{num_epochs} | train Loss: {train_loss:.4f} | train Accuracy: {train_acc:.4f} | valid Loss: {val_loss:.4f} | valid Accuracy: {val_acc:.4f} | test Accuracy: {test_acc:.4f} | patient: {patient}")

        # Early Stopping Test
        save_dir = f"/mnt/data2/DSAINet/weight/{config['model']['name']}/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_val_model_state, f"/mnt/data2/DSAINet/weight/{config['model']['name']}/{args.dataset}/{test_fold_id}.pth")
        
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
        all_val_fold_acc.append(acc)
        # kappa
        kappa = cohen_kappa_score(all_targets, all_preds)
        all_val_fold_kappa.append(kappa)
        # f1
        f1 = f1_score(all_targets, all_preds, average='weighted')
        all_val_fold_f1.append(f1)
        logger.info(f"Early Stopping Best Epoch: {best_val_epoch} | Test Fold {test_fold_id + 1} | Accuracy: {acc:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

        # All Epochs Test
        save_dir = f"/mnt/data2/DSAINet/weight/{config['model']['name']}/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_test_model_state, f"/mnt/data2/DSAINet/weight/{config['model']['name']}/{args.dataset}/best_{test_fold_id}.pth")
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
        all_test_fold_acc.append(acc)
        # kappa
        kappa = cohen_kappa_score(all_targets, all_preds)
        all_test_fold_kappa.append(kappa)
        # f1
        f1 = f1_score(all_targets, all_preds, average='weighted')
        all_test_fold_f1.append(f1)
        logger.info(f"All Best Epoch: {best_test_epoch} | Test Fold {test_fold_id + 1} | Accuracy: {acc:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

    logger.info(f"========== {fold_num}-Fold Done ==========")

    logger.info("----- Early Stopping Results -----")
    for i, (acc, kappa, f1) in enumerate(zip(all_val_fold_acc, all_val_fold_kappa, all_val_fold_f1)):
        logger.info(f"Fold {i+1:02d} | Acc = {acc:.4f} | Kappa = {kappa:.4f} | F1 = {f1:.4f}")

    logger.info("----- All Epoch Results -----")
    for i, (acc, kappa, f1) in enumerate(zip(all_test_fold_acc, all_test_fold_kappa, all_test_fold_f1)):
        logger.info(f"Fold {i+1:02d} | Acc = {acc:.4f} | Kappa = {kappa:.4f} | F1 = {f1:.4f}")

    logger.info(f"Early Stopping - Average KFold Accuracy: {np.mean(all_val_fold_acc):.4f}±{np.std(all_val_fold_acc):.4f}")
    logger.info(f"Early Stopping - Average KFold Kappa: {np.mean(all_val_fold_kappa):.4f}±{np.std(all_val_fold_kappa):.4f}")
    logger.info(f"Early Stopping - Average KFold F1: {np.mean(all_val_fold_f1):.4f}±{np.std(all_val_fold_f1):.4f}")

    logger.info(f"All Epoch - Average KFold Accuracy: {np.mean(all_test_fold_acc):.4f}±{np.std(all_test_fold_acc):.4f}")
    logger.info(f"All Epoch - Average KFold Kappa: {np.mean(all_test_fold_kappa):.4f}±{np.std(all_test_fold_kappa):.4f}")
    logger.info(f"All Epoch - Average KFold F1: {np.mean(all_test_fold_f1):.4f}±{np.std(all_test_fold_f1):.4f}")

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
    
    # epoch
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    
    # batch size
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    
    # learning rate
    if args.lr is not None:
        config['train']['lr'] = args.lr

    # logger
    logger = setup_logger(f"{args.dataset}", log_dir=f"./log/{config['model']['name']}/", overwrite=True)
    # device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # log config 
    logger.info("==== Experiment Config ====")
    logger.info(yaml.dump(config).rstrip())
    logger.info("==========================")

    # Load data
    data, labels, n_class, n_channels, n_times = load_data(args.dataset, 'KFold')

    train_test_kfold(data, labels, config, device, logger)