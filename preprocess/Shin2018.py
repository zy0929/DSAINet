import os
import re, warnings
from typing import Iterable, List, Tuple

import numpy as np
import mne
from scipy.io import loadmat

mne.set_log_level("ERROR")
warnings.filterwarnings(
    "ignore",
    message="Channels contain different lowpass filters.*",
    category=RuntimeWarning,
    module="mne"
)

# rest = 20s before anchor, task = 20s after anchor
ANCHOR_CODE = 48
SEG_LEN_S = 20.0    # segment length for rest and task

root_dir = "/mnt/data2/DSAINet/Shin2018"
target_channels = [
    "FP1", "AFF5", "AFz", "F1", "FC5", "FC1", "T7",
    "C3", "Cz", "CP5", "CP1", "P7", "P3", "Pz", 
    "POz", "O1", "FP2", "AFF6", "F2", "FC2", "FC6", 
    "C4", "T8", "CP2", "CP6", "P4", "P8", "O2"
]

# Raw source bandpass
bandpass = (0.5, 40.0)
resample_rate = 250.0

def _subject_folder(subject_id: int) -> str:
    return f"VP{subject_id:03d}"

def _read_raw_session(subject_dir: str, stem: str) -> mne.io.BaseRaw:
    vhdr_path = os.path.join(subject_dir, f"{stem}.vhdr")
    if not os.path.exists(vhdr_path):
        raise FileNotFoundError(f"Missing file: {vhdr_path}")
    return mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

def _preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    present = [ch for ch in target_channels if ch in raw.ch_names]
    missing = [ch for ch in target_channels if ch not in present]
    if missing:
        print(f"[WARN] Missing channels: {missing}")

    raw.pick_channels(present, ordered=False)
    raw.reorder_channels(present)

    raw.filter(bandpass[0], bandpass[1], picks="eeg", verbose=False)
    raw.resample(resample_rate, npad="auto", verbose=False)
    return raw

def _extract_anchors_from_annotations(raw: mne.io.BaseRaw, code: int = ANCHOR_CODE) -> np.ndarray:
    """
    BrainVision markers become MNE annotations.
    Match descriptions like 'Stimulus/S 48', 'S 48', etc.
    """
    if raw.annotations is None or len(raw.annotations) == 0:
        return np.array([], dtype=int)

    pat = re.compile(rf"\bS\s*{int(code)}\b")

    anchors = []
    for onset_s, desc in zip(raw.annotations.onset, raw.annotations.description):
        if pat.search(str(desc)):
            anchors.append(int(raw.time_as_index(onset_s, use_rounding=True)[0]))

    return np.asarray(sorted(anchors), dtype=int) if anchors else np.array([], dtype=int)

def _extract_task_rest_windows(raw: mne.io.BaseRaw, anchors: np.ndarray, win_len_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each anchor sample t:
        REST: [t-20s, t)
        TASK: [t, t+20s)
    Split each 20s segment into 4s non-overlapping windows (5 windows each).
    Labels: rest=0, task=1
    Output: X (N, C, T), y (N,)
    """
    sfreq = float(raw.info["sfreq"])
    x = raw.get_data()  # (C, samples), Volts

    seg_samp = int(round(SEG_LEN_S * sfreq))
    win_samp = int(round(win_len_s * sfreq))
    n_win = seg_samp // win_samp  # 20/4=5

    n_ch, n_time = x.shape
    X_list, y_list = [], []

    def add_segment(seg_start: int, label: int):
        seg_end = seg_start + seg_samp
        if seg_start < 0 or seg_end > n_time:
            return
        for k in range(n_win):
            a = seg_start + k * win_samp
            b = a + win_samp
            if b <= n_time:
                X_list.append(x[:, a:b])
                y_list.append(label)

    for t in anchors:
        add_segment(t - seg_samp, label=0)  # rest
        add_segment(t, label=1)            # task

    if not X_list:
        return (
            np.empty((0, n_ch, win_samp), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y

def process_one_dsr_session(subject_dir: str, stem: str, win_len_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Read -> preprocess -> anchors -> task/rest windows for one session file."""
    raw = _read_raw_session(subject_dir, stem)
    raw = _preprocess_raw(raw)
    anchors = _extract_anchors_from_annotations(raw, code=ANCHOR_CODE)
    return _extract_task_rest_windows(raw, anchors, win_len_s)

def load_kfold_shin2018(
    subject_ids: Iterable[int] = range(1, 27),
    session_stems: Tuple[str, str, str] = ("gonogo1", "gonogo2", "gonogo3"),
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    LOSO-style data container: returns per-subject arrays,
    each subject = concat of gonogo1/2/3 after preprocessing & segmentation.
    """
    all_subject_data, all_subject_labels = [], []
    win_len_s = 4.0

    for sid in subject_ids:
        subj_dir = os.path.join(root_dir, _subject_folder(sid))
        if not os.path.isdir(subj_dir):
            raise FileNotFoundError(f"Missing subject folder: {subj_dir}")

        X_parts, y_parts = [], []
        for stem in session_stems:
            X, y = process_one_dsr_session(subj_dir, stem, win_len_s)
            X_parts.append(X)
            y_parts.append(y)

        X_subj = np.concatenate(X_parts, axis=0) if X_parts else np.empty(
            (0, len(target_channels), int(win_len_s * resample_rate)), dtype=np.float32
        )
        y_subj = np.concatenate(y_parts, axis=0) if y_parts else np.empty((0,), dtype=np.int64)

        all_subject_data.append(X_subj)
        all_subject_labels.append(y_subj)

    return all_subject_data, all_subject_labels