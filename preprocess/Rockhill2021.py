import os
import numpy as np
import mne

# -----------------------
# Basic settings 
# -----------------------
mode = "ON"  # HC vs PDON. Alternative: "OFF", HC vs PDOFF  

mne.set_log_level(verbose="ERROR")
root_dir = "/mnt/data2/DSAINet/Rockhill2021"  

target_channels = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]  

bandpass = (0.5, 40.0)
resample_rate = 250.0
epoch_len = 4.0  # seconds
overlap = 2.0  # seconds

subjects_not_existing = {15, 27}
subject_list = [x for x in range(1, 34) if x not in subjects_not_existing] 

def get_file_and_label(subject_id: int, test_mode: str = mode, root: str = root_dir):
    """
    output: (file_path, label_str) where label_str in {"H","PD"}
    """
    hc_name = f"sub-hc{subject_id}"
    hc_dir = os.path.join(root, hc_name)
    if os.path.isdir(hc_dir):
        label = "H"
        name = hc_name
        file_path = os.path.join(
            root, name, "ses-hc", "eeg", f"{name}_ses-hc_task-rest_eeg.bdf"
        )
    else:
        label = "PD"
        name = f"sub-pd{subject_id}"
        if test_mode == "ON":
            file_path = os.path.join(
                root, name, "ses-on", "eeg", f"{name}_ses-on_task-rest_eeg.bdf"
            )
        elif test_mode == "OFF":
            file_path = os.path.join(
                root, name, "ses-off", "eeg", f"{name}_ses-off_task-rest_eeg.bdf"
            )
        else:
            raise ValueError("Mode should be either 'ON' or 'OFF'.")

    return file_path, label


def _fixed_length_epochs_from_raw(
    raw: mne.io.BaseRaw,
    epoch_len: float = 2.0,
    overlap: float = 1.0,
):
    """
    Resting-state: create events by cutting the continuous signal into fixed windows.
    Returns an mne.Epochs object.
    """
    sfreq = raw.info["sfreq"]
    # ensure last epoch fits fully
    stop = raw.times[-1] - epoch_len + (1.0 / sfreq)
    if stop <= 0:
        raise ValueError(f"Recording too short for epoch_len={epoch_len}s")

    events = mne.make_fixed_length_events(
        raw,
        id=1,
        start=0.0,
        stop=stop,
        duration=epoch_len,
        overlap=overlap,
    )
    event_id = {"rest": 1}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=epoch_len - 1.0 / sfreq,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )
    return epochs


def load_kfold_rockhill2021(test_mode: str = mode, epoch_len: float = epoch_len, overlap: float = overlap):
    """
    Returns:
      all_subject_data   : list of np.ndarray, each (n_epochs, 32, n_times)
      all_subject_labels : list of np.ndarray, each (n_epochs,) with HC=0, PD=1
    """
    all_subject_data = []
    all_subject_labels = []

    for subject_id in subject_list:
        file_path, label_str = get_file_and_label(subject_id, test_mode=test_mode)

        raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)

        # Unit handling (BioSemi is read in Volts by MNE)
        raw._data *= 1e6

        # Normal preprocessing stages
        raw.pick_channels(target_channels, ordered=True)
        raw.filter(bandpass[0], bandpass[1], verbose=False)
        raw.resample(sfreq=resample_rate, npad="auto", verbose=False)

        # Fixed-length epoching for resting-state 
        epochs = _fixed_length_epochs_from_raw(raw, epoch_len=epoch_len, overlap=overlap)
        data = epochs.get_data().astype(np.float32)  # (n_epochs, 32, n_times)

        # Labels: HC=0, PD=1 (binary) 
        y = 0 if label_str == "H" else 1
        labels = np.full((data.shape[0],), y, dtype=np.int64)

        all_subject_data.append(data)
        all_subject_labels.append(labels)

    return all_subject_data, all_subject_labels