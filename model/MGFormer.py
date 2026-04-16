import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple

def get_sinusoidal_position_encoding(seq_len: int, dim: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe


# similar to Deformers attention block
class MultiHeadAttention(nn.Module):
    """
    A standard multi-head self-attention mechanism.
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        project_out = not (heads == 1 and dim_head == dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PositionWiseFeedForward(nn.Module):
    """
    A position-wise feed-forward network.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, fine_grained_kernel: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.feed_forward = PositionWiseFeedForward(dim, mlp_dim, dropout)
        self.cnn_block = self._create_cnn_block(dim, fine_grained_kernel, dropout)

    def _create_cnn_block(self, in_dim: int, kernel_size: int, dropout: float) -> nn.Module:
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(in_dim),
            nn.ELU()
        )

    def _get_fft_feature(self, x: torch.Tensor) -> torch.Tensor:
        fft_magnitude = torch.fft.rfft(x, dim=-1).abs()
        return torch.log(torch.mean(fft_magnitude, dim=-1) + 1e-8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The input x is expected to have shape (batch_size, embed_dim, seq_len)
        x_t = x.transpose(1, 2)
        attention_out = self.attention(x_t).transpose(1, 2)
        x_coarse = attention_out + x

        x_fine = self.cnn_block(x)
        fft_feat = self._get_fft_feature(x_fine)

        ff_out = self.feed_forward(x_coarse.transpose(1, 2)).transpose(1, 2)
        x_out = ff_out + x_fine

        return x_out, fft_feat

class TransformerEncoder(nn.Module):

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, fine_grained_kernel: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, fine_grained_kernel, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_features = []
        for layer in self.layers:
            x, fft_feat = layer(x)
            dense_features.append(fft_feat)

        x_pooled = torch.mean(x, dim=-1)
        x_dense = torch.cat(dense_features, dim=-1)
        return torch.cat([x_pooled, x_dense], dim=-1)

class MultiGranularityTokenEncoder(nn.Module):
    def __init__(self, num_channels: int, sampling_rate: int, embed_dim: int, num_T: int, dropout_rate: float = 0.1):
        super().__init__()
        granularities = [0.04, 0.06, 0.08]
        kernel_sizes = [int(sampling_rate * g) for g in granularities]
        self.pool_size = 4

        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_T, kernel_size=(1, ks), stride=(1, max(1, ks - 1))),
                nn.LeakyReLU()
            ) for ks in kernel_sizes
        ])
        self.bn_t = nn.BatchNorm2d(num_T)

        self.spatial_conv = nn.Conv2d(num_T, embed_dim, kernel_size=(num_channels, 1), stride=1)
        self.activation = nn.LeakyReLU()
        self.bn_s = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor, pool: int = None) -> torch.Tensor:
        pool_k = pool if pool is not None else self.pool_size
        x = x.unsqueeze(1)  # (B, 1, C, T)

        branch_outputs = []
        for branch in self.temporal_branches:
            out = branch(x)
            out = F.max_pool2d(out, kernel_size=(1, pool_k), stride=(1, pool_k))
            branch_outputs.append(out)

        combined = torch.cat(branch_outputs, dim=-1)
        combined = self.bn_t(combined)

        spatial_out = self.spatial_conv(combined)
        spatial_out = self.activation(spatial_out)
        spatial_out = F.max_pool2d(spatial_out, kernel_size=(1, max(1, pool_k // 4)))
        spatial_out = self.bn_s(spatial_out)

        return spatial_out.squeeze(2).permute(0, 2, 1)

class MGFormer(nn.Module):
    def __init__(self, *, num_chan: int, num_time: int, sampling_rate: int, embed_dim: int,
                 num_classes: int, num_T: int, depth: int = 4, heads: int = 16,
                 mlp_dim: int = 16, dim_head: int = 16, dropout: float = 0.5,
                 fine_grained_kernel: int = 11):
        super().__init__()
        self.token_encoder = MultiGranularityTokenEncoder(
            num_channels=num_chan,
            sampling_rate=sampling_rate,
            embed_dim=embed_dim,
            num_T=num_T,
            dropout_rate=dropout
        )

        self.transformer_encoder = TransformerEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fine_grained_kernel=fine_grained_kernel,
            dropout=dropout
        )

        self.mlp_head = nn.Linear(embed_dim * (depth + 1), num_classes)

    def forward(self, x: torch.Tensor, pool: int = None) -> torch.Tensor:
        x = x.squeeze(1)  # (B, C, T)
        tokens = self.token_encoder(x, pool=pool)
        B, L, D = tokens.shape

        pos_emb = get_sinusoidal_position_encoding(L, D, device=x.device)
        tokens = tokens + pos_emb.unsqueeze(0)

        tokens = tokens.permute(0, 2, 1)
        transformer_out = self.transformer_encoder(tokens)

        return self.mlp_head(transformer_out)

if __name__ == '__main__':
    # Example usage
    model = MGFormer(
        num_chan=22,
        num_time=1000,
        sampling_rate=250,
        embed_dim=128,
        num_classes=4,
        num_T=64,
        depth=4,
        heads=8,
        mlp_dim=256,
        dim_head=16,
        dropout=0.5,
        fine_grained_kernel=11
    )

    dummy_input = torch.randn(16, 22, 1000)
    output = model(dummy_input)
    print("Model Architecture:\n", model)
    print("\nOutput shape:", output.shape)
