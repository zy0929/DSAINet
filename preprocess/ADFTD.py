import numpy as np
import pandas as pd
import mne
import os
import pickle

'''
Dataset: OpenNeuro ds004504 (Alzheimer's Disease EEG)
Type: Resting-state (Closed Eyes)
Subjects: 88 (36 AD, 23 FTD, 29 CN)
Sample rate: 500Hz (downsampled to 200Hz in code)
Channels: 19 Scalp EEG channels (10-20 system)

Labels Mapping:
0: CN (Healthy Control)
1: AD (Alzheimer's disease)
2: FTD (Frontotemporal Dementia)

Strategy:
Since this is resting-state data (continuous), we slide a window of 4 seconds
to create "epochs" artificially.
'''

mne.set_log_level(verbose='ERROR') 
root_dir = '/mnt/data2/DSAINet/ADFTD' 

# Preprocessing
bandpass = (0.5, 45.0) 
resample_rate = 250.0 
epoch_window = (0.0, 4.0)

preprocess_info = {
    "bandpass": list(bandpass),
    "resample": resample_rate,
    "tmin": epoch_window[0],
    "tmax": epoch_window[1],
}

# Load Subject Info & Labels
participants_path = os.path.join(root_dir, 'participants.tsv')
df_participants = pd.read_csv(participants_path, sep='\t')

# 'sub-001' -> 'CN'
sub_to_group = dict(zip(df_participants['participant_id'], df_participants['Group']))

# label mapping
label_map = {'C': 0, 'A': 1, 'F': 2}

subject_trial_counter = {}

# 19 channel
target_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                   'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

def load_kfold_adftd():
    all_subject_data = [] 
    all_subject_labels = []
    for sub_id_str in df_participants['participant_id']:
        subject_id_num = int(sub_id_str.split('-')[1]) 
        
        group_str = sub_to_group[sub_id_str]
            
        label_idx = label_map[group_str]
        subject_trial_counter.setdefault(subject_id_num, 0)

        # Data Loading
        file_path = os.path.join(root_dir, 'derivatives', sub_id_str, 'eeg', f'{sub_id_str}_task-eyesclosed_eeg.set')

        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        # uv
        raw._data *= 1e6

        # Preprocessing
        raw.pick_channels(target_channels)
        raw.set_montage('standard_1020')
        # Bandpass Filter
        raw.filter(preprocess_info['bandpass'][0], preprocess_info['bandpass'][1])
        # Resample
        raw.resample(preprocess_info['resample'])

        # Epoching (Sliding Window)
        events = mne.make_fixed_length_events(raw, id=1, start=0, duration=4.0)
        epochs = mne.Epochs(raw, events, event_id=1, 
                            tmin=0, tmax=4.0 - 1.0/raw.info['sfreq'], 
                            baseline=None, preload=True, verbose=False)
        
        data = epochs.get_data() # (n_epochs, n_channels, n_times)
        n_epochs = data.shape[0]
        labels = np.full((n_epochs,), int(label_idx), dtype=np.int64)

        all_subject_data.append(data)
        all_subject_labels.append(labels)

    return all_subject_data, all_subject_labels