import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure reproducibility with cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_name, log_dir="./log", overwrite=False):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger

    if overwrite and os.path.exists(log_path):
        os.remove(log_path)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

def classwise_augmentation(data: torch.Tensor,
                                 label: torch.Tensor,
                                 n_segments: int = 8):
    """
    Segmentation & Reconstruction (Torch version)

    Args:
        data : Tensor, shape (B, 1, C, T), can be on CPU or GPU
        label: Tensor, shape (B,), dtype long/int, can be on CPU or GPU
        n_segments: int, number of segments (default 8)

    Returns:
        train_data : Tensor, shape (B + B_aug, 1, C, T)
        train_label: Tensor, shape (B + B_aug,)
    """
    # Ensure tensor types
    if not torch.is_tensor(data):
        raise TypeError("data must be a torch.Tensor")
    if not torch.is_tensor(label):
        raise TypeError("label must be a torch.Tensor")

    device = data.device
    dtype = data.dtype
    label = label.to(device)

    B, _, C, T = data.shape
    classes = torch.unique(label)
    n_classes = int(classes.numel())

    # Edge cases
    if n_classes == 0:
        return data, label
    n_aug = B // n_classes
    if n_aug == 0:
        return data, label

    seg_len = T // n_segments
    if seg_len == 0:
        return data, label

    aug_data_list = []
    aug_label_list = []

    # For each class, build augmented samples
    for cls in classes:
        cls_idx = (label == cls).nonzero(as_tuple=True)[0]
        cls_data = data.index_select(0, cls_idx)  # (Nc,1,C,T)
        Nc = cls_data.shape[0]
        if Nc == 0:
            continue

        # Random indices: (n_aug, n_segments), each entry in [0, Nc)
        rand_idx = torch.randint(low=0, high=Nc, size=(n_aug, n_segments), device=device)

        tmp_aug = torch.zeros((n_aug, 1, C, T), device=device, dtype=dtype)
        for seg in range(n_segments):
            start = seg * seg_len
            end = (seg + 1) * seg_len
            if start >= T:
                break
            end = min(end, T)

            picked = cls_data.index_select(0, rand_idx[:, seg])  # (n_aug,1,C,T)
            tmp_aug[..., start:end] = picked[..., start:end]

        tmp_lab = torch.full((n_aug,), cls, device=device, dtype=label.dtype)

        aug_data_list.append(tmp_aug)
        aug_label_list.append(tmp_lab)

    if len(aug_data_list) == 0:
        return data, label

    aug_data = torch.cat(aug_data_list, dim=0)
    aug_label = torch.cat(aug_label_list, dim=0)

    # Shuffle aug
    perm = torch.randperm(aug_data.shape[0], device=device)
    aug_data = aug_data.index_select(0, perm)
    aug_label = aug_label.index_select(0, perm)

    # Concat original + aug
    train_data = torch.cat([data, aug_data], dim=0)
    train_label = torch.cat([label, aug_label], dim=0)

    return train_data, train_label
