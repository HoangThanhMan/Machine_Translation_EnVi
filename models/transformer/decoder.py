import torch
import torch.nn as nn

from models.transformer.self_attention import MultiHeadAttentionLayer
from models.transformer.feed_forward import FeedForward
from models.transformer.positional_encoder import PositionalEncoder


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.enc_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.ffn = FeedForward(hid_dim, pf_dim, dropout)

        self.layer_norm1 = nn.LayerNorm(hid_dim)
        self.layer_norm2 = nn.LayerNorm(hid_dim)
        self.layer_norm3 = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_mask):
        # Masked self-attention
        _x, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout(_x))

        # Encoder-decoder attention
        _x, attn = self.enc_attn(x, memory, memory, src_mask)
        x = self.layer_norm2(x + self.dropout(_x))

        # Feedforward
        _x = self.ffn(x)
        x = self.layer_norm3(x + self.dropout(_x))

        return x, attn


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()

        self.device = device
        self.token_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(hid_dim, dtype=torch.float32)).to(device)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        # tgt: [batch size, tgt len]
        batch_size, tgt_len = tgt.shape

        tok_emb = self.token_embedding(tgt) * self.scale  # [batch, tgt_len, hid_dim]

        pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)  # [batch, tgt_len]
        pos_emb = self.pos_embedding(pos)  # [batch, tgt_len, hid_dim]

        x = self.dropout(tok_emb + pos_emb)  # [batch, tgt_len, hid_dim]

        attn = None
        for layer in self.layers:
            x, attn = layer(x, memory, tgt_mask, src_mask)

        output = self.fc_out(x)  # [batch, tgt_len, output_dim]
        return output, attn
