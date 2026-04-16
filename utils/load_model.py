import numpy as np
import torch
import torch.nn as nn
from model.EEGNet import EEGNet
from model.Conformer import Conformer
from model.ShallowConvNet import ShallowConvNet
from model.DeepConvNet import DeepConvNet
from model.CTNet import CTNet
from model.ADFCNN import ADFCNN
from model.DBConformer import DBConformer
from model.MSCFormer import MSCFormer
from model.MSVTNet import MSVTNet
from model.TMSANet import TMSANet
from model.Deformer import Deformer
from model.LMDANet import LMDANet
from model.MGFormer import MGFormer
from model.DSAINet import DSAINet

def load_model(model_params, config, n_class, n_channels, n_times, logger, device):
    # EEGNet
    if model_params['name'] == 'EEGNet':
        model = EEGNet(
            n_classes=n_class,
            Chans=n_channels,
            Samples=n_times,
            kernLenght=model_params['kernLenght'],
            F1=model_params['F1'],
            D=model_params['D'],
            F2=model_params['F2'],
            dropoutRate=model_params['dropoutRate'],
            norm_rate=model_params['norm_rate']
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)

    # Conformer
    elif model_params['name'] == 'Conformer':
        model = Conformer(
            args=model_params,
            emb_size=model_params['emb_size'],
            depth=model_params['depth'],
            chn=n_channels,
            n_classes=n_class,
            dropout=model_params['dropout'],
            drop_p=model_params['drop_p'],
            forward_drop_p=model_params['forward_drop_p'],
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)

    # ShallowConvNet
    elif model_params['name'] == 'ShallowConvNet':
        model = ShallowConvNet(
            nChan=n_channels,
            nTime=n_times,
            nClass=n_class,
            dropoutP=model_params['dropoutRate']
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay) 
    
    # DeepConvNet
    elif model_params['name'] == 'DeepConvNet':
        model = DeepConvNet(
            nChan=n_channels,
            nTime=n_times,
            nClass=n_class,
            dropoutP=model_params['dropoutRate']
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay) 

    # CTNet
    elif model_params['name'] == 'CTNet':
        model = CTNet(
            heads = model_params['heads'],
            emb_size = model_params['emb_size'],
            depth = model_params['depth'],
            eeg1_f1 = model_params['eeg1_f1'],
            eeg1_kernel_size = model_params['eeg1_kernel_size'],
            eeg1_D = model_params['eeg1_D'],
            eeg1_pooling_size1 = model_params['eeg1_pooling_size1'],
            eeg1_pooling_size2 = model_params['eeg1_pooling_size2'],
            eeg1_dropout_rate = model_params['eeg1_dropout_rate'],
            flatten_eeg1 = model_params['flatten_eeg1'],
            number_class = n_class,
            eeg1_number_channel = n_channels
        ).to(device)
        b1 = config['train']['b1']
        b2 = config['train']['b2']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], betas=(b1, b2))
    
    # ADFCNN
    elif model_params['name'] == 'ADFCNN':
        model = ADFCNN(
            num_classes = n_class,
            num_channels = n_channels
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)

    # LMDANet
    if model_params['name'] == 'LMDANet':
        model = LMDANet(
            num_classes=n_class,
            chans=n_channels,
            samples=n_times,
            depth=model_params['depth'],
            kernel=model_params['kernel'],
            channel_depth1=model_params['channel_depth1'],
            channel_depth2=model_params['channel_depth2'],
            ave_depth=model_params['ave_depth'],
            avepool=model_params['avepool']
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay) 

    # DBConformer
    elif model_params['name'] == 'DBConformer':
        model = DBConformer(
            args=model_params['args'],
            emb_size=model_params['emb_size'],
            tem_depth=model_params['tem_depth'],
            chn_depth=model_params['chn_depth'],
            n_classes=n_class,
            chn=n_channels
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)
    
    # MSCFormer
    elif model_params['name'] == 'MSCFormer':
        model = MSCFormer(
            class_num=n_class,
            dropout_rate=model_params['dropout_rate'],
            chn=n_channels
        ).to(device)
        b1 = config['train']['b1']
        b2 = config['train']['b2']
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], betas=(b1, b2), weight_decay=weight_decay)

    # MSVTNet
    elif model_params['name'] == 'MSVTNet':
        model = MSVTNet(
            nCh=n_channels,
            nTime=1000,
            cls=n_class,
            F=model_params['F'],
            C1=model_params['C1'],
            C2=model_params['C2'],
            D=model_params['D'],
            P1=model_params['P1'],
            P2=model_params['P2'],
            Pc=model_params['Pc'],
            nhead=model_params['nhead'],
            ff_ratio=model_params['ff_ratio'],
            Pt=model_params['Pt'],
            layers=model_params['layers']
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)

    # TMSANet
    elif model_params['name'] == 'TMSANet':
        model = TMSANet(
            in_planes=n_channels,
            radix=model_params['radix'],
            time_points=1000,
            num_classes=n_class,
            embed_dim=model_params['embed_dim'],
            pool_size=model_params['pool_size'],
            pool_stride=model_params['pool_stride'],
            num_heads=model_params['num_heads'],
            fc_ratio=model_params['fc_ratio'],
            depth=model_params['depth'],
            attn_drop=model_params['attn_drop'],
            fc_drop=model_params['fc_drop']
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)
    
    # MGFormer
    elif model_params['name'] == 'MGFormer':
        model = MGFormer(
            num_chan=n_channels,               
            num_time=n_times,                 
            sampling_rate=200,
            embed_dim=model_params['embed_dim'],
            num_classes=n_class,
            num_T=model_params['num_T'],
            depth=model_params['depth'],
            heads=model_params['heads'],
            mlp_dim=model_params['mlp_dim'],
            dim_head=model_params['dim_head'],
            dropout=model_params['dropout'],
            fine_grained_kernel=model_params['fine_grained_kernel'],
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)
    
    # Deformer
    elif model_params['name'] == 'Deformer':
        model = Deformer(
            num_chan=n_channels,
            num_time=n_times,
            temporal_kernel=model_params['temporal_kernel'],
            num_kernel=model_params['num_kernel'],
            num_classes=n_class,
            depth=model_params['depth'],
            heads=model_params['heads'],
            mlp_dim=model_params['mlp_dim'],
            dim_head=model_params['dim_head'],
            dropout=model_params['dropoutRate']
        ).to(device)
        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)

    # DSAINet
    elif model_params['name'] == 'DSAINet':
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

            self_ffn_expansion=model_params['self_ffn_expansion'],
            cross_ffn_expansion=model_params['cross_ffn_expansion'],

            big_residual=model_params['big_residual'],
            big_residual_learnable=model_params['big_residual_learnable'],

            cls_dropout=model_params['cls_dropout']
        ).to(device)

        weight_decay = config['train']['weight_decay']
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=weight_decay)

    return model, criterion, optimizer

      
