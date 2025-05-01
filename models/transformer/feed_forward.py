import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, dff), # (d_model, dff) -> (d_model, dff)
            nn.ReLU(),            # (dff, d_model) -> (d_model, d_model)
            nn.Linear(dff, d_model), # (d_model, d_model)
            nn.Dropout(dropout_rate) # (d_model, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.layer_norm(x + self.seq(x)) # (d_model, d_model)
        return x
