import numpy as np
import mne
import os
import pickle
import scipy.io as sio

root_dir = '/mnt/data2/DSAINet/OpenBMI/'

mne.set_log_level(verbose='ERROR') 

target_channels = [
    'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
]

standard_channels = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 
    'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 
    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 
    'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 
    'FT9', 'FTT 9h', 'TPP 7h', 'TP7', 'TPP 9h', 'FT10', 'FTT 10h', 'TPP 8h', 'TP8', 'TPP 10h', 
    'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'
]

preprocess_info = {
    "bandpass": [0.5, 40.0],
    "resample": 250.0,
    "tmin": 0.0,
    "tmax": 4.0,
    "channels": target_channels
}

def load_kfold_openbmi():
    all_subject_data = [] 
    all_subject_labels = []
    for subject_id in range(1, 55):
        subject_epochs = []
        subject_labels = []
        for session_id in [1, 2]:
            dir_name = f'session{session_id}'
            file_name = f'sess{session_id:02d}_subj{subject_id:02d}_EEG_MI.mat'
            data_path = os.path.join(root_dir, dir_name, file_name)

            mat_content = sio.loadmat(data_path)

            # read train and test data
            train = mat_content['EEG_MI_train']
            test = mat_content['EEG_MI_test']

            train_raw_data = train[0, 0]['x']
            train_raw_data = train_raw_data.T
            train_raw_label = train[0, 0]['y_dec']
            train_triggers = train[0, 0]['t']
            train_original_fs = train[0, 0]['fs'][0, 0]

            test_raw_data = test[0, 0]['x']
            test_raw_data = test_raw_data.T
            test_raw_label = test[0, 0]['y_dec']
            test_triggers = test[0, 0]['t']
            test_original_fs = test[0, 0]['fs'][0, 0]

            train_triggers = np.array(train_triggers).squeeze().astype(int)
            train_labels = np.array(train_raw_label).squeeze().astype(int) - 1
            test_triggers = np.array(test_triggers).squeeze().astype(int)
            test_labels = np.array(test_raw_label).squeeze().astype(int) - 1

            # Create MNE RawArray objects
            info = mne.create_info(ch_names=standard_channels, sfreq=float(train_original_fs), ch_types='eeg')
            raw_train = mne.io.RawArray(train_raw_data, info)
            raw_test = mne.io.RawArray(test_raw_data, info)

            # Pick target channels
            raw_train.pick_channels(preprocess_info['channels'])
            raw_test.pick_channels(preprocess_info['channels'])

            # epoching
            train_n_events = min(len(train_triggers), len(train_labels))
            train_events = np.column_stack((
                    train_triggers[:train_n_events], 
                    np.zeros(train_n_events, dtype=int), 
                    train_labels[:train_n_events]
                ))
            test_n_events = min(len(test_triggers), len(test_labels))
            test_events = np.column_stack((
                    test_triggers[:test_n_events], 
                    np.zeros(test_n_events, dtype=int), 
                    test_labels[:test_n_events]
                ))
            event_id = {'Right': 0, 'Left': 1}
            
            epochs_train = mne.Epochs(raw_train, train_events, event_id=event_id,
                                    tmin=preprocess_info['tmin'], tmax=preprocess_info['tmax']-1.0/raw_train.info['sfreq'], baseline=None, preload=True)
            epochs_test = mne.Epochs(raw_test, test_events, event_id=event_id,
                                    tmin=preprocess_info['tmin'], tmax=preprocess_info['tmax']-1.0/raw_test.info['sfreq'], baseline=None, preload=True)

            # bandpass filter
            epochs_train.filter(preprocess_info['bandpass'][0], preprocess_info['bandpass'][1])
            epochs_test.filter(preprocess_info['bandpass'][0], preprocess_info['bandpass'][1])
            # resample
            epochs_train.resample(preprocess_info['resample'])
            epochs_test.resample(preprocess_info['resample'])

            train_data = epochs_train.get_data()  # (n_epochs, n_channels, n_times)
            train_label = epochs_train.events[:, -1]  # (n_epochs,)
            test_data = epochs_test.get_data()  # (n_epochs, n_channels, n_times)
            test_label = epochs_test.events[:, -1]  # (n_epochs,)

            data = np.concatenate((train_data, test_data), axis=0)
            label = np.concatenate((train_label, test_label), axis=0)

            subject_epochs.append(data)
            subject_labels.append(label)
        
        subject_epochs = np.concatenate(subject_epochs, axis=0)
        subject_labels = np.concatenate(subject_labels, axis=0)

        all_subject_data.append(subject_epochs)
        all_subject_labels.append(subject_labels)
    
    return all_subject_data, all_subject_labels

'''
total_samples: 21600
sessions: {'sess02', 'sess01'}
subjects: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54}
labels: {0: 10800, 1: 10800}
shapes: {(20, 1000)}
'''