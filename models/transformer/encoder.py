import torch
import torch.nn as nn

from models.transformer.self_attention import MultiHeadAttentionLayer
from models.transformer.feed_forward import FeedForward
from models.transformer.positional_encoder import PositionalEncoder

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.layer_norm1 = nn.LayerNorm(hid_dim)
        self.layer_norm2 = nn.LayerNorm(hid_dim)
        self.feed_forward = FeedForward(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # Self-attention sublayer
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.layer_norm1(src + self.dropout(_src))  # Residual + LayerNorm

        # Position-wise feed-forward sublayer
        _src = self.feed_forward(src)
        src = self.layer_norm2(src + self.dropout(_src))  # Residual + LayerNorm

        return src


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=2048):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_encoder = PositionalEncoder(hid_dim, max_length, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(hid_dim, dtype=torch.float32)).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        # Token embedding
        src = self.tok_embedding(src) * self.scale  # [batch size, src len, hid_dim]

        # Positional encoding
        src = self.pos_encoder(src)  # [batch size, src len, hid_dim]

        # Pass through stacked encoder layers
        for layer in self.layers:
            src = layer(src, src_mask)

        return src  # [batch size, src len, hid_dim]
