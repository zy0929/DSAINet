import numpy as np
import pandas as pd
import mne
import os

mne.set_log_level(verbose='ERROR') 
root_dir = '/mnt/data2/DSAINet/EEGMat/'

selected_channels = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz']

bandpass = (0.5, 45.0) 
resample_rate = 250.0 
epoch_window = (0.0, 2.0) 

preprocess_info = {
    "bandpass": list(bandpass),
    "resample": resample_rate,
    "tmin": epoch_window[0],
    "tmax": epoch_window[1],
}

def load_kfold_eegmat():
    all_subject_data = [] 
    all_subject_labels = []
    for subject_id in range(36):
        subject_epochs = []
        subject_labels = []
        for file_type, label in [("_1", 0), ("_2", 1)]:
            file_name = f"Subject{subject_id:02d}{file_type}.edf"
            file_path = os.path.join(root_dir, file_name)
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            # uv
            raw._data *= 1e6

            # pick selected channels
            raw.pick_channels(selected_channels)
            # bandpass filter
            raw.filter(bandpass[0], bandpass[1])
            # resample
            # raw.resample(resample_rate)

            # ---- crop to 60s (AFTER resample) ----
            total_dur = raw.times[-1]  # seconds, last time point

            if file_type == "_1":
                # last 60s
                tmin_crop = max(0.0, total_dur - 60.0)
                tmax_crop = total_dur
                raw.crop(tmin=tmin_crop, tmax=tmax_crop, include_tmax=True)
            else:  # "_2"
                # first 60s
                tmin_crop = 0.0
                tmax_crop = min(60.0, total_dur)
                raw.crop(tmin=tmin_crop, tmax=tmax_crop, include_tmax=True)

            events = mne.make_fixed_length_events(raw, id=1, duration=2.0, overlap=1.0)
            event_id = {'segment': 1}
            # epoching
            epochs = mne.Epochs(raw, events = events, event_id = event_id,
                                tmin = epoch_window[0], tmax = epoch_window[1]-1.0/raw.info['sfreq'],
                                baseline = None, preload = True)
            # save data and labels
            data = epochs.get_data()  # (n_trials, n_channels, n_times)
            labels = np.full(data.shape[0], fill_value=label, dtype=int)

            subject_epochs.append(data)
            subject_labels.append(labels)
        
        subject_epochs = np.concatenate(subject_epochs, axis=0)
        subject_labels = np.concatenate(subject_labels, axis=0)

        all_subject_data.append(subject_epochs)
        all_subject_labels.append(subject_labels)

    return all_subject_data, all_subject_labels
