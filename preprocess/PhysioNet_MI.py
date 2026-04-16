import numpy as np
import pandas as pd
import mne
import os
import pickle

'''
sampling rate: 160Hz
electrodes: 64 channels(10-10 system)

1 baseline: eyes open
2 baseline: eyes closed
3 7 11 task 1: open and close left or right fist
4 8 12 task 2: imagine opening and closing left or right fist
5 9 13 task 3: open and close both fists or both feet
6 10 14 task 4: imagine opening and closing both fists or both feet

T0 corresponds to rest
T1 corresponds to onset of motion (real or imagined) of
- the left fist (in runs 3, 4, 7, 8, 11, and 12)
- both fists (in runs 5, 6, 9, 10, 13, and 14)
T2 corresponds to onset of motion (real or imagined) of
- the right fist (in runs 3, 4, 7, 8, 11, and 12)
- both feet (in runs 5, 6, 9, 10, 13, and 14)
'''

mne.set_log_level(verbose='ERROR') 
tasks = [4, 6, 8, 10, 12, 14]  # motor imagery
root_dir = '/mnt/data2/DSAINet/PhysioNet_MI/'

bandpass = (0.5, 40.0)
resample_rate = 250.0
epoch_window = (0.0, 4.0)
preprocess_info = {
    "bandpass": list(bandpass),
    "resample": resample_rate,
    "tmin": epoch_window[0],
    "tmax": epoch_window[1],
}

def load_kfold_physionet_mi():
    all_subject_data = [] 
    all_subject_labels = []

    for subject_id in range(1, 110):
        subject_epochs = []
        subject_labels = []
        for task in tasks:
            file = os.path.join(root_dir, f'S{subject_id:03d}', f'S{subject_id:03d}R{task:02d}.edf')
            raw = mne.io.read_raw_edf(file, preload=True)
            # convert to uV
            raw._data *= 1e6
            # bandpass filter
            raw.filter(preprocess_info['bandpass'][0], preprocess_info['bandpass'][1])
            # resample
            raw.resample(sfreq=preprocess_info['resample'], npad="auto")
            # epoching
            events, event_id = mne.events_from_annotations(raw)
            epochs = mne.Epochs(
                raw,
                events,
                event_id,
                tmin=preprocess_info['tmin'],
                tmax=preprocess_info['tmax'] - 1.0/raw.info['sfreq'],
                baseline=None,
                preload=True
            )
            data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            label = epochs.events[:, -1]  # (n_epochs,)

            '''
            0: left fist
            1: right fist
            2: both fists
            3: both feet
            '''
            for i, (data_idx, label_idx) in enumerate(zip(data, label)):
                # label remapping
                if label_idx == 1:
                    continue
                elif label_idx == 2:
                    if task in [4, 8, 12]:
                        label_idx = 0
                    elif task in [6, 10, 14]:
                        label_idx = 2
                elif label_idx == 3:
                    if task in [4, 8, 12]:
                        label_idx = 1
                    elif task in [6, 10, 14]:
                        label_idx = 3
                label_idx = int(label_idx)

                subject_epochs.append(data_idx)
                subject_labels.append(label_idx)

        subject_epochs = np.array(subject_epochs, dtype=np.float32)
        subject_labels = np.array(subject_labels, dtype=np.int64)

        all_subject_data.append(subject_epochs)
        all_subject_labels.append(subject_labels)

    return all_subject_data, all_subject_labels
            
'''
total_samples: 9837
sessions: {4, 6, 8, 10, 12, 14}
subjects: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109}
labels: {0: 2479, 1: 2438, 2: 2465, 3: 2455}
shapes: {(64, 1000)}
'''
