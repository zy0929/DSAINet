import os
import mne
import numpy as np
import pickle

'''
Mumtaz2017:
- Dataset: 34 (MDD) + 30 (HC)
- Electrodes: 19 channels, international 10-20 system
- Sampling rate: 256Hz
- Paradigm: EO (Eyes open), EC (Eyes closed), Task (SSVEP, shall be eliminated)
'''

selected_channels = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                     'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
                     'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE']

root_dir = '/mnt/data2/DSAINet/Mumtaz2017'

bandpass = (0.3, 75.0)
resample_rate = 200.0
epoch_window = (0.0, 5.0)
preprocess_info = {
    "bandpass": list(bandpass),
    "resample": resample_rate,
    "tmin": epoch_window[0],
    "tmax": epoch_window[1],
}

mne.set_log_level(verbose='WARNING') 

def load_kfold_mumtaz2017():
    all_subject_data = [] 
    all_subject_labels = []
    # HC
    for subject_id in range(1, 31):
        subject_epochs = []
        subject_labels = []
        for session_id in ['EC', 'EO']:
            if subject_id == 15 and session_id == 'EO':
                file_name = f'6921143_H S{subject_id} {session_id}.edf'
                new_data_path = os.path.join(root_dir, file_name)
                raw = mne.io.read_raw_edf(new_data_path, preload = True)
                # uv
                raw._data *= 1e6
                # pick selected channels
                raw.pick_channels(selected_channels)
                # bandpass filter
                raw.filter(bandpass[0], bandpass[1])
                # notch filter at 50 Hz
                raw.notch_filter(freqs=50)
                # resample
                raw.resample(resample_rate)
                # Generate synthetic events every 4s
                n_samples = raw.n_times
                segment_len = int(preprocess_info['resample'] * (preprocess_info['tmax'] - preprocess_info['tmin']))
                n_segments = n_samples // segment_len

                # Create fake events at start of each 4s segment
                events = np.array([[i * segment_len, 0, 1] for i in range(n_segments)])
                event_id = {'segment': 1}
                # epoching
                epochs = mne.Epochs(raw, events = events, event_id = event_id,
                                    tmin = epoch_window[0], tmax = epoch_window[1]-1.0/raw.info['sfreq'],
                                    baseline = None, preload = True)
                # save data and labels
                data = epochs.get_data()  # (n_trials, n_channels, n_times)
                HC_labels = np.zeros(data.shape[0], dtype=int)  # HC label 0

                subject_epochs.append(data)
                subject_labels.append(HC_labels)

            if subject_id == 15 and session_id == 'EO':
                file_name = f'6921959_H S{subject_id} {session_id}.edf'
                new_data_path = os.path.join(root_dir, file_name)
                raw = mne.io.read_raw_edf(new_data_path, preload = True)
                # uv
                raw._data *= 1e6
                # pick selected channels
                raw.pick_channels(selected_channels)
                # bandpass filter
                raw.filter(bandpass[0], bandpass[1])
                # notch filter at 50 Hz
                raw.notch_filter(freqs=50)
                # resample
                raw.resample(resample_rate)
                # Generate synthetic events every 4s
                n_samples = raw.n_times
                segment_len = int(preprocess_info['resample'] * (preprocess_info['tmax'] - preprocess_info['tmin']))
                n_segments = n_samples // segment_len

                # Create fake events at start of each 4s segment
                events = np.array([[i * segment_len, 0, 1] for i in range(n_segments)])
                event_id = {'segment': 1}
                # epoching
                epochs = mne.Epochs(raw, events = events, event_id = event_id,
                                    tmin = epoch_window[0], tmax = epoch_window[1]-1.0/raw.info['sfreq'],
                                    baseline = None, preload = True)
                # save data and labels
                data = epochs.get_data()  # (n_trials, n_channels, n_times)
                HC_labels = np.zeros(data.shape[0], dtype=int)  # HC label 0

                subject_epochs.append(data)
                subject_labels.append(HC_labels)

            # read raw data
            file_name = f'H S{subject_id} {session_id}.edf'
            data_path = os.path.join(root_dir, file_name)
            if not os.path.exists(data_path):
                continue
            raw = mne.io.read_raw_edf(data_path, preload = True)
            # uv
            raw._data *= 1e6
            # pick selected channels
            raw.pick_channels(selected_channels)
            # bandpass filter
            raw.filter(bandpass[0], bandpass[1])
            # notch filter at 50 Hz
            raw.notch_filter(freqs=50)
            # resample
            raw.resample(resample_rate)
            # Generate synthetic events every 4s
            n_samples = raw.n_times
            segment_len = int(preprocess_info['resample'] * (preprocess_info['tmax'] - preprocess_info['tmin']))
            n_segments = n_samples // segment_len

            # Create fake events at start of each 4s segment
            events = np.array([[i * segment_len, 0, 1] for i in range(n_segments)])
            event_id = {'segment': 1}
            # epoching
            epochs = mne.Epochs(raw, events = events, event_id = event_id,
                                tmin = epoch_window[0], tmax = epoch_window[1]-1.0/raw.info['sfreq'],
                                baseline = None, preload = True)
            # save data and labels
            data = epochs.get_data()  # (n_trials, n_channels, n_times)
            HC_labels = np.zeros(data.shape[0], dtype=int)  # HC label 0

            subject_epochs.append(data)
            subject_labels.append(HC_labels)

        if len(subject_epochs) >= 2:
            subject_data = np.concatenate(subject_epochs, axis=0)
            subject_label = np.concatenate(subject_labels, axis=0)
        elif len(subject_epochs) == 1:
            subject_data = subject_epochs[0]
            subject_label = subject_labels[0]
        else:
            continue

        all_subject_data.append(subject_data)
        all_subject_labels.append(subject_label)

    # MDD
    for subject_id in range(1, 35):
        subject_epochs = []
        subject_labels = []
        for session_id in ['EC', 'EO']:
            # read raw data
            file_name = f'MDD S{subject_id} {session_id}.edf'
            data_path = os.path.join(root_dir, file_name)
            if not os.path.exists(data_path):
                continue
            raw = mne.io.read_raw_edf(data_path, preload = True)
            # uv
            raw._data *= 1e6
            # pick selected channels
            raw.pick_channels(selected_channels)
            # bandpass filter
            raw.filter(bandpass[0], bandpass[1])
            # notch filter at 50 Hz
            raw.notch_filter(freqs=50)
            # resample
            raw.resample(resample_rate)
            # Generate synthetic events every 4s
            n_samples = raw.n_times
            segment_len = int(preprocess_info['resample'] * (preprocess_info['tmax'] - preprocess_info['tmin']))
            n_segments = n_samples // segment_len
            # Create fake events at start of each 4s segment
            events = np.array([[i * segment_len, 0, 1] for i in range(n_segments)])
            event_id = {'segment': 1}
            # epoching  
            epochs = mne.Epochs(raw, events = events, event_id = event_id,
                                tmin = epoch_window[0], tmax = epoch_window[1]-1.0/raw.info['sfreq'],
                                baseline = None, preload = True)
            # save data and labels
            data = epochs.get_data()  # (n_trials, n_channels, n_times)
            MDD_labels = np.ones(data.shape[0], dtype=int)  # MDD label 1

            subject_epochs.append(data)
            subject_labels.append(MDD_labels)

        if len(subject_epochs) >= 2:
            subject_data = np.concatenate(subject_epochs, axis=0)
            subject_label = np.concatenate(subject_labels, axis=0)
        elif len(subject_epochs) == 1:
            subject_data = subject_epochs[0]
            subject_label = subject_labels[0]
        else:
            continue

        all_subject_data.append(subject_data)
        all_subject_labels.append(subject_label)

    return all_subject_data, all_subject_labels
