import numpy as np
import torch
import torch.nn as nn
from model.DSAINet import DSAINet

def load_model(model_params, config, n_class, n_channels, n_times, logger, device):
    # DSAINet
    if model_params['name'] == 'DSAINet':
        model = DSAINet(
            n_classes=n_class,
            Chans=n_channels,
            Samples=n_times,

            emb_size=model_params['emb_size'],
            heads=model_params['heads'],
            attn_depth=model_params['attn_depth'],
            attn_dropout=model_params['attn_dropout'],

            eeg1_f1=model_params['eeg1_f1'],
            eeg1_kernel_size=model_params['eeg1_kernel_size'],
            eeg1_D=model_params['eeg1_D'],
            eeg1_pooling_size1=model_params['eeg1_pooling_size1'],
            eeg1_pooling_size2=model_params['eeg1_pooling_size2'],
            eeg1_dropout_rate=model_params['eeg1_dropout_rate'],

            branch_1_kernels=model_params['branch_1_kernels'],
            branch_2_kernels=model_params['branch_2_kernels'],
            conv_expansion=model_params['conv_expansion'],
            conv_dropout=model_params['conv_dropout'],

            intra_ffn_expansion=model_params['intra_ffn_expansion'],
            inter_ffn_expansion=model_params['inter_ffn_expansion'],

            big_residual=model_params['big_residual'],
            big_residual_learnable=model_params['big_residual_learnable'],

            cls_dropout=model_params['cls_dropout']
        ).to(device)

        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)
    
    return model, criterion, optimizer

      