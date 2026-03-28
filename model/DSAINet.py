'''
@File:      DSAINet.py
@Author:    
@Function:  Original model code of "DSAINet: An Efficient Dual-Scale Attentive Interaction Network for General EEG Decoding"
'''

import math
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np


# -----------------------------
# patch embedding
# -----------------------------
class PatchEmbedding(nn.Module):
    """
    Input : (B, 1, C, T)
    Output: (B, f2, 1, N)
    """
    def __init__(
        self,
        f1=16, kernel_size=64, D=2,
        pooling_size1=4, pooling_size2=8,
        dropout_rate=0.25,
        number_channel=22
    ):
        super().__init__()
        f2 = D * f1
        self.f2 = f2

        self.net = nn.Sequential(
            # (B,1,C,T) -> (B,f1,C,T)
            nn.Conv2d(1, f1, (1, kernel_size), padding="same", bias=False),
            nn.BatchNorm2d(f1),

            # (B,f1,C,T) -> (B,f2,1,T)   (kernel height = C, valid padding collapses C->1)
            nn.Conv2d(f1, f2, (number_channel, 1), groups=f1, padding="valid", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # (B,f2,1,T) -> (B,f2,1,T1)
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),

            # (B,f2,1,T1) -> (B,f2,1,T1)
            nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # (B,f2,1,T1) -> (B,f2,1,N)
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,f2,1,N)

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    Input/Output: (B, N, E)
    """
    def __init__(self, emb_size: int, length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, length, emb_size))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,E)
        n = x.shape[1]
        return self.drop(x + self.pe[:, :n, :].to(x.device))

# -----------------------------
# ConvTime layer
# -----------------------------
class ConvTimeLayer(nn.Module):
    """
    Input/Output: (B, E, N)
    - Depthwise Conv1d over N (groups=E)  -> (B,E,N)
    - Pointwise FFN (1x1 conv)            -> (B,E,N)
    - BN + residual (learnable alpha inside the layer)
    """
    def __init__(self, emb_size: int, kernel_size: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(
            emb_size, emb_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=emb_size,
            bias=False
        )
        d_ff = expansion * emb_size
        self.pw1 = nn.Conv1d(emb_size, d_ff, 1, groups=4, bias=False)
        self.pw2 = nn.Conv1d(d_ff, emb_size, 1, groups=4, bias=False)
        self.bn = nn.BatchNorm1d(emb_size)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        # inside-layer residual scaling
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,E,N)
        y = self.act(self.dw(x))     # (B,E,N)
        y = self.act(self.pw1(y))    # (B,4E,N)
        y = self.pw2(y)              # (B,E,N)
        y = self.bn(y)               # (B,E,N)
        y = self.drop(y)
        return x + self.alpha * y    # (B,E,N)

class ConvTimeStack(nn.Module):
    """
    A stack of ConvTimeLayer, one per kernel in kernel_list.

    Input/Output: (B,E,N)
    """
    def __init__(self, emb_size: int, kernel_list: List[int], expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([ConvTimeLayer(emb_size, k, expansion, dropout) for k in kernel_list])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,E,N)
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------
# Attention blocks
# -----------------------------
class IntraAttnBlock(nn.Module):
    def __init__(self, emb_size: int, heads: int, dropout: float = 0.1, ffn_expansion: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.drop = nn.Dropout(dropout)

        d_ff = ffn_expansion * emb_size
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, emb_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,E)
        attn_out, _ = self.mha(x, x, x)              # (B,N,E)
        x = self.norm1(x + self.drop(attn_out))      # (B,N,E)
        x = self.norm2(x + self.drop(self.ffn(x)))   # (B,N,E)
        return x

class InterAttnBlock(nn.Module):
    def __init__(self, emb_size, heads, dropout=0.1, ffn_expansion=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_size, heads, dropout=dropout, batch_first=True)

        self.norm1a = nn.LayerNorm(emb_size)
        self.norm1b = nn.LayerNorm(emb_size)
        self.norm2a = nn.LayerNorm(emb_size)
        self.norm2b = nn.LayerNorm(emb_size)

        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn  = nn.Dropout(dropout)

        # learnable injection strengths
        self.beta12 = nn.Parameter(torch.tensor(1.0))
        self.beta21 = nn.Parameter(torch.tensor(1.0))

        d_ff = ffn_expansion * emb_size
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, emb_size),
        )

    def forward(self, x1, x2):
        out1, _ = self.mha(x1, x2, x2)
        y1 = self.norm1a(x1 + self.drop_attn(self.beta12 * out1))
        y1 = self.norm1b(y1 + self.drop_ffn(self.ffn(y1)))

        out2, _ = self.mha(x2, x1, x1)
        y2 = self.norm2a(x2 + self.drop_attn(self.beta21 * out2))
        y2 = self.norm2b(y2 + self.drop_ffn(self.ffn(y2)))

        return y1, y2

# -----------------------------
# 2-branch EEG model
# -----------------------------
class DSAINet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        Chans: int,
        Samples: int,
        emb_size: int = 40,
        heads: int = 4,
        attn_depth: int = 1,
        attn_dropout: float = 0.25,

        # patch embedding
        eeg1_f1: int = 16,
        eeg1_kernel_size: int = 64,
        eeg1_D: int = 2,
        eeg1_pooling_size1: int = 4,
        eeg1_pooling_size2: int = 8,
        eeg1_dropout_rate: float = 0.25,

        # branch kernel lists
        branch_1_kernels: Optional[List[int]] = [11, 15],
        branch_2_kernels: Optional[List[int]] = [3, 7],
        conv_expansion: int = 4,
        conv_dropout: float = 0.25,

        # intra-attention
        intra_ffn_expansion: int = 2,
        # inter-attention
        inter_ffn_expansion: int = 2,

        # big residual controls
        big_residual: bool = True,
        big_residual_learnable: bool = True,

        # classifier
        cls_dropout: float = 0.25,
    ):
        super().__init__()

        # defaults requested
        if branch_1_kernels is None:
            branch_1_kernels = [19, 19]
        if branch_2_kernels is None:
            branch_2_kernels = [29, 29]

        self.emb_size = emb_size
        self.attn_depth = attn_depth
        self.big_residual = big_residual

        pos_len = Samples // (eeg1_pooling_size1 * eeg1_pooling_size2)

        # Patch embedding map: (B,1,C,T)->(B,f2,1,N)
        self.patch = PatchEmbedding(
            f1=eeg1_f1, kernel_size=eeg1_kernel_size, D=eeg1_D,
            pooling_size1=eeg1_pooling_size1, pooling_size2=eeg1_pooling_size2,
            dropout_rate=eeg1_dropout_rate, number_channel=Chans
        )
        f2 = eeg1_f1 * eeg1_D

        # tokens: (B,N,f2)->(B,N,E)
        self.proj = nn.Linear(f2, emb_size) if f2 != emb_size else nn.Identity()
        self.pos = PositionalEncoding(emb_size, length=pos_len, dropout=attn_dropout)

        # Two parallel ConvTime stacks (both take the SAME a0 in (B,E,N) form)
        self.branch1 = ConvTimeStack(emb_size, branch_1_kernels, expansion=conv_expansion, dropout=conv_dropout)
        self.branch2 = ConvTimeStack(emb_size, branch_2_kernels, expansion=conv_expansion, dropout=conv_dropout)

        # Big residual scalars
        if big_residual:
            if big_residual_learnable:
                self.alpha1 = nn.Parameter(torch.tensor(1.0))
                self.alpha2 = nn.Parameter(torch.tensor(1.0))
            else:
                # fixed = 1
                self.register_buffer("alpha1", torch.tensor(1.0), persistent=False)
                self.register_buffer("alpha2", torch.tensor(1.0), persistent=False)

        # Intra-attn blocks for each branch (separate)
        self.intra_1 = nn.ModuleList([IntraAttnBlock(emb_size, heads, attn_dropout, intra_ffn_expansion) for _ in range(attn_depth)])
        self.intra_2 = nn.ModuleList([IntraAttnBlock(emb_size, heads, attn_dropout, intra_ffn_expansion) for _ in range(attn_depth)])

        # Inter-attn blocks (symmetric)
        self.inter = nn.ModuleList([InterAttnBlock(emb_size, heads, attn_dropout, inter_ffn_expansion) for _ in range(attn_depth)])

        # Classifier
        self.token_attn = nn.Linear(emb_size, 1)  # shared across branches
        self.classifier = nn.Sequential(
            nn.Dropout(cls_dropout),
            nn.Linear(2 * emb_size, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,1,C,T)
        """
        B = x.shape[0]

        # Patch embedding map: (B,f2,1,N)
        fmap = self.patch(x)

        # a0 tokens: (B,N,E)
        a0 = fmap.squeeze(2).transpose(1, 2)      # (B,N,f2)
        a0 = self.proj(a0)                        # (B,N,E)
        a0 = a0 * math.sqrt(self.emb_size)
        a0 = self.pos(a0)                         # (B,N,E)

        # Parallel ConvTime branches (operate on (B,E,N))
        z0 = a0.transpose(1, 2)                   # (B,E,N)
        a1 = self.branch1(z0).transpose(1, 2)     # (B,N,E)
        a2 = self.branch2(z0).transpose(1, 2)     # (B,N,E)

        # Big residual injection from a0
        if self.big_residual:
            a1 = a1 + self.alpha1 * a0
            a2 = a2 + self.alpha2 * a0

        # Intra-attn + inter-attn
        for i in range(self.attn_depth):
            a1 = self.intra_1[i](a1)               # (B,N,E)
            a2 = self.intra_2[i](a2)               # (B,N,E)
            a1, a2 = self.inter[i](a1, a2)         # (B,N,E), (B,N,E)

        # Token attention pooling + classification
        x = torch.stack([a1, a2], dim=1)          # (B,2,N,E)
        w = torch.softmax(self.token_attn(x).squeeze(-1), dim=2)  # (B,2,N)
        pooled = (x * w.unsqueeze(-1)).sum(dim=2)  # (B,2,E)

        feat = pooled.reshape(B, -1)              # (B,2E) = 80 when E=40
        return self.classifier(feat)
