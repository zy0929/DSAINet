import mne
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from preprocess.BCIC_IV_2a import load_loso_bcic_iv_2a
from preprocess.BCIC_IV_2b import load_loso_bcic_iv_2b
from preprocess.Zhou2016 import load_loso_zhou2016
from preprocess.PhysioNet_MI import load_kfold_physionet_mi
from preprocess.OpenBMI import load_kfold_openbmi
from preprocess.Mumtaz2017 import load_kfold_mumtaz2017
from preprocess.ADFTD import load_kfold_adftd
from preprocess.EEGMat import load_kfold_eegmat
from preprocess.Shin2018 import load_kfold_shin2018
from preprocess.Rockhill2021 import load_kfold_rockhill2021

def load_data(dataset_name, strategy):
    # BCIC-IV-2a
    if dataset_name == 'BCIC-IV-2a' and strategy == 'LOSO':
        data, labels = load_loso_bcic_iv_2a()
        n_class = 4
        n_channels = 22
        n_times = 1000
    # BCIC-IV-2b
    elif dataset_name == 'BCIC-IV-2b' and strategy == 'LOSO':
        data, labels = load_loso_bcic_iv_2b()
        n_class = 2
        n_channels = 3
        n_times = 1000
    # Zhou2016
    elif dataset_name == 'Zhou2016' and strategy == 'LOSO':
        data, labels = load_loso_zhou2016()
        n_class = 3
        n_channels = 14
        n_times = 1000
    # PhysioNet-MI
    elif dataset_name == 'PhysioNet-MI' and strategy == 'KFold':
        data, labels = load_kfold_physionet_mi()
        n_class = 4
        n_channels = 64
        n_times = 1000
    # OpenBMI
    elif dataset_name == 'OpenBMI' and strategy == 'KFold':
        data, labels = load_kfold_openbmi()
        n_class = 2
        n_channels = 20
        n_times = 1000
    # Mumtaz2017
    elif dataset_name == 'Mumtaz2017' and strategy == 'KFold':
        data, labels = load_kfold_mumtaz2017()
        n_class = 2
        n_channels = 19
        n_times = 1000
    # ADFTD
    elif dataset_name == 'ADFTD' and strategy == 'KFold':
        data, labels = load_kfold_adftd()
        n_class = 3
        n_channels = 19
        n_times = 1000
    # EEGMat
    elif dataset_name == 'EEGMat' and strategy == 'KFold':
        data, labels = load_kfold_eegmat()
        n_class = 2
        n_channels = 19
        n_times = 1000
    # Shin2018
    elif dataset_name == 'Shin2018' and strategy == 'KFold':
        data, labels = load_kfold_shin2018()
        n_class = 2
        n_channels = 28
        n_times = 1000
    # Rockhill2021
    elif dataset_name == 'Rockhill2021' and strategy == 'KFold':
        data, labels = load_kfold_rockhill2021()
        n_class = 2
        n_channels = 32
        n_times = 1000
    
    return data, labels, n_class, n_channels, n_times