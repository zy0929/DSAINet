import os
from pathlib import Path
import numpy as np
import mne
import pickle

mne.set_log_level("ERROR")
removed_channels = ['VEOU', 'VEOL']
root_dir = "/mnt/data2/DSAINet/Zhou2016"          

bandpass = (0.5, 40.0)          # paper-aligned MI band
resample_rate = 200.0           # 200Hz to align Sample
tmin, tmax = 1.0, 6.0           # cue/motor imagery window (1s prep then 5s imagery)

preprocess_info = {
    "bandpass": list(bandpass),
    "resample": resample_rate,
    "tmin": tmin,
    "tmax": tmax,
}

def load_loso_zhou2016():
    # class 1 left, class 2 right, class 3 foot
    event_id = {"left": 1, "right": 2, "foot": 3}
    ann_to_code = {"1": 1, "2": 2, "3": 3, "255": 255}

    all_subject_data = []
    all_subject_labels = []

    for subj_dir in sorted(Path(root_dir).glob("subject_*")):
        subject_epochs = []
        subject_labels = []

        for cnt_path in sorted(subj_dir.glob("*.cnt")):
            raw = mne.io.read_raw_cnt(str(cnt_path), preload=True)

            raw.pick_types(eeg=True, eog=False)
            # convert to uV
            raw._data *= 1e6
            # drop EOG
            to_drop = [i for i in removed_channels if i in raw.ch_names]
            raw.drop_channels(to_drop)
            # filter
            raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])
            # resample
            raw.resample(resample_rate)

            events, _ = mne.events_from_annotations(raw, event_id=ann_to_code)
            events = events[np.isin(events[:, 2], [1, 2, 3])]

            # epoching
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax - 1.0 / raw.info["sfreq"],
                baseline=None,
                preload=True,
                reject_by_annotation=True,
                verbose=False,
            )

            X = epochs.get_data().astype(np.float32, copy=False)
            y = epochs.events[:, 2].astype(int) - 1  # 0/1/2

            subject_epochs.append(X)
            subject_labels.append(y)

        subject_epochs = np.concatenate(subject_epochs, axis=0)
        subject_labels = np.concatenate(subject_labels, axis=0)

        all_subject_data.append(subject_epochs)
        all_subject_labels.append(subject_labels)

    return all_subject_data, all_subject_labels

'''
total_samples: 1800
sessions: {1, 2, 3}
subjects: {1, 2, 3, 4}
labels: {0: 600, 1: 599, 2: 601}
shapes: {(14, 1000)}

suitable: loso & cs
'''