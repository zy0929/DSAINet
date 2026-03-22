"""
util/norm.py
- Subject-level normalization utilities shared by data loaders.
"""

from collections import defaultdict
import numpy as np
import torch


class SubjectZScoreTransform:
    def __init__(self, subject_stats, global_stats, eps=1e-6):
        self.subject_stats = {int(k): (v[0].clone(), v[1].clone()) for k, v in subject_stats.items()}
        self.global_stats = (global_stats[0].clone(), global_stats[1].clone()) if global_stats else None
        self.eps = eps

    def __call__(self, sample):
        subj = int(sample["subject"])
        x = sample["x"].float()
        mean_std = self.subject_stats.get(subj)
        if mean_std is None:
            if self.global_stats is not None:
                mean_std = (self.global_stats[0], self.global_stats[1])
            else:
                mean = x.mean(dim=0, keepdim=True)
                std = torch.clamp(x.std(dim=0, keepdim=True), min=self.eps)
                mean_std = (mean, std)
            self.subject_stats[subj] = (mean_std[0].clone(), mean_std[1].clone())
        mean, std = mean_std
        std = std.clamp_min(self.eps)
        normed = (x - mean) / std
        sample = dict(sample)
        sample["x"] = normed.to(sample["x"].dtype)
        return sample


class SubjectEAStatic:
    """Applies precomputed EA matrices on the training set."""

    def __init__(self, subject_mats):
        self.subject_mats = {
            int(k): torch.as_tensor(v, dtype=torch.float32).clone()
            for k, v in (subject_mats or {}).items()
        }

    def __call__(self, sample):
        subj = int(sample["subject"])
        mat = self.subject_mats.get(subj)
        if mat is None:
            return sample
        x = sample["x"].float().squeeze(0)
        aligned = torch.matmul(mat, x)
        sample = dict(sample)
        sample["x"] = aligned.unsqueeze(0).to(sample["x"].dtype)
        return sample


class GlobalEAStatic:
    """Applies a single precomputed EA matrix to all subjects."""

    def __init__(self, mat):
        self.mat = torch.as_tensor(mat, dtype=torch.float32).clone() if mat is not None else None

    def __call__(self, sample):
        if self.mat is None:
            return sample
        x = sample["x"].float().squeeze(0)
        aligned = torch.matmul(self.mat, x)
        sample = dict(sample)
        sample["x"] = aligned.unsqueeze(0).to(sample["x"].dtype)
        return sample


class SubjectEASequentialPrecomputed:
    """Sequential EA that reuses cumulative covariances for validation/test splits."""

    def __init__(self, dataset, eps=1e-6):
        self.lookup = {}
        self.eps = eps
        self._prepare(dataset)

    def _prepare(self, dataset):
        by_subj = defaultdict(list)
        orders = getattr(dataset, "trial_order", None)
        for idx in range(len(dataset)):
            subj = int(dataset.subjects[idx])
            order_val = float(orders[idx]) if orders is not None else float(idx)
            order_key = int(round(order_val))
            by_subj[subj].append((order_key, idx))
        for subj, seq in by_subj.items():
            seq.sort(key=lambda t: (t[0], t[1]))
            ref = None
            count = 0
            for order_key, idx in seq:
                sample = dataset[idx]
                x = sample["x"].float().squeeze(0).numpy()
                cov = np.cov(x) + self.eps * np.eye(x.shape[0], dtype=np.float32)
                if ref is None:
                    ref = cov
                else:
                    ref = (ref * count + cov) / (count + 1)
                count += 1
                mat = _matrix_power_neg_half(ref, self.eps)
                self.lookup[(int(subj), order_key)] = torch.from_numpy(mat)

    def __call__(self, sample):
        subj = int(sample["subject"])
        order_tensor = sample.get("order")
        if isinstance(order_tensor, torch.Tensor):
            order_key = int(round(float(order_tensor.item())))
        else:
            order_key = int(round(float(order_tensor)))
        mat = self.lookup.get((subj, order_key))
        if mat is None:
            return sample
        x = sample["x"].float().squeeze(0)
        aligned = torch.matmul(mat, x)
        sample = dict(sample)
        sample["x"] = aligned.unsqueeze(0).to(sample["x"].dtype)
        return sample


def _compute_subject_stats(dataset, eps=1e-6):
    if len(dataset) == 0:
        return {}, None
    sums, sumsq, counts = {}, {}, {}
    global_sum = None
    global_sumsq = None
    total = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        x = sample["x"].float()
        subj = int(sample["subject"])
        if subj not in sums:
            sums[subj] = torch.zeros_like(x)
            sumsq[subj] = torch.zeros_like(x)
            counts[subj] = 0
        sums[subj] += x
        sumsq[subj] += x * x
        counts[subj] += 1
        if global_sum is None:
            global_sum = torch.zeros_like(x)
            global_sumsq = torch.zeros_like(x)
        global_sum += x
        global_sumsq += x * x
        total += 1
    stats = {}
    for subj, total_sum in sums.items():
        mean = total_sum / counts[subj]
        var = sumsq[subj] / counts[subj] - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_min(eps)
        stats[int(subj)] = (mean, std)
    if total == 0:
        return stats, None
    global_mean = global_sum / total
    global_var = global_sumsq / total - global_mean * global_mean
    global_std = torch.sqrt(torch.clamp(global_var, min=0.0)).clamp_min(eps)
    return stats, (global_mean, global_std)


def _matrix_power_neg_half(matrix, eps=1e-6):
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, eps, None).astype(np.float64, copy=False)
    inv_sqrt = np.diag(vals ** -0.5)
    result = vecs @ inv_sqrt @ vecs.T
    return result.astype(np.float32)


def _compute_subject_ea(dataset, eps=1e-6):
    if dataset is None or len(dataset) == 0:
        return {}
    cov_sums, counts = {}, {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        x = sample["x"].float().squeeze(0).numpy()
        subj = int(sample["subject"])
        cov = np.cov(x) + eps * np.eye(x.shape[0], dtype=np.float32)
        if subj not in cov_sums:
            cov_sums[subj] = cov
            counts[subj] = 1
        else:
            cov_sums[subj] += cov
            counts[subj] += 1
    subject_mats = {
        int(subj): _matrix_power_neg_half(cov_sums[subj] / counts[subj], eps)
        for subj in cov_sums
    }
    return subject_mats


def _compute_global_ea(dataset, eps=1e-6):
    if dataset is None or len(dataset) == 0:
        return None
    cov_sum = None
    count = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        x = sample["x"].float().squeeze(0).numpy()
        cov = np.cov(x) + eps * np.eye(x.shape[0], dtype=np.float32)
        if cov_sum is None:
            cov_sum = cov
        else:
            cov_sum += cov
        count += 1
    if count == 0:
        return None
    return _matrix_power_neg_half(cov_sum / count, eps)


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _split_epochs(x):
    x = _as_numpy(x).astype(np.float32, copy=False)
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0, :, :], True
    if x.ndim == 3:
        return x, False
    if x.ndim == 4:
        return x, False
    raise ValueError(f"Unsupported data shape: {x.shape}")


def _apply_ea_matrix_to_array(data, mat):
    if data is None:
        return None
    x, add_dim = _split_epochs(data)
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        out[i] = mat @ x[i]
    if add_dim:
        out = out[:, None, :, :]
    return out


def _apply_online_ea_array(data, eps=1e-6):
    if data is None:
        return None
    x, add_dim = _split_epochs(data)
    out = np.empty_like(x, dtype=np.float32)
    ref = None
    count = 0
    for i in range(x.shape[0]):
        cov = np.cov(x[i]) + eps * np.eye(x.shape[1], dtype=np.float32)
        if ref is None:
            ref = cov
        else:
            ref = (ref * count + cov) / (count + 1)
        count += 1
        mat = _matrix_power_neg_half(ref, eps)
        out[i] = mat @ x[i]
    if add_dim:
        out = out[:, None, :, :]
    return out


def _mean_covariance(data, eps=1e-6):
    x, _ = _split_epochs(data)
    cov_sum = None
    count = 0
    for i in range(x.shape[0]):
        cov = np.cov(x[i]) + eps * np.eye(x.shape[1], dtype=np.float32)
        if cov_sum is None:
            cov_sum = cov
        else:
            cov_sum += cov
        count += 1
    if count == 0:
        return None
    return cov_sum / count


def apply_ea_to_arrays(train_data, valid_data, test_data, n_channels, eps=1e-6):
    """
    Array-based EA for train/valid/test splits (no subject metadata required).
    - Train: static EA matrix from training set mean covariance.
    - Valid/Test: online EA with sequential cumulative covariance (per split).
    """
    if train_data is None:
        return train_data, valid_data, test_data
    train_x, _ = _split_epochs(train_data)
    if train_x.shape[1] != n_channels:
        raise ValueError(f"n_channels mismatch: got {train_x.shape[1]}, expected {n_channels}")
    mean_cov = _mean_covariance(train_data, eps=eps)
    if mean_cov is None:
        return train_data, valid_data, test_data
    mat = _matrix_power_neg_half(mean_cov, eps)
    train_aligned = _apply_ea_matrix_to_array(train_data, mat)
    valid_aligned = _apply_online_ea_array(valid_data, eps=eps)
    test_aligned = _apply_online_ea_array(test_data, eps=eps)
    return train_aligned, valid_aligned, test_aligned
