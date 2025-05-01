import torch
import torch.nn as nn
import math

def positional_encoding(length, depth):
    depth = depth // 2

    positions = torch.arange(length).unsqueeze(1).float()  # (seq_len, 1)
    depths = torch.arange(depth).unsqueeze(0).float() / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (seq_len, depth)

    pos_encoding = torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], dim=-1)  # (seq_len, depth*2)

    return pos_encoding  # (length, depth*2)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = positional_encoding(max_len, d_model)
        
    def forward(self, x):
        length = x.size(1)
        x = x * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:length, :].unsqueeze(0).to(x.device)
        x = self.dropout(x)
        return x
