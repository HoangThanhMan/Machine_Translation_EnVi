import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim) # [batch size, query len, hid dim]
        self.fc_k = nn.Linear(hid_dim, hid_dim) # [batch size, key len, hid dim]
        self.fc_v = nn.Linear(hid_dim, hid_dim) # [batch size, value len, hid dim]

        self.fc_o = nn.Linear(hid_dim, hid_dim) 

        self.dropout = nn.Dropout(dropout) 

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        # [batch size, n heads, query len, head dim]
        Q = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # [batch size, n heads, key len, head dim]
        K = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # [batch size, n heads, value len, head dim]
        V = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # [batch size, n heads, query len, key len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim = -1)

        # [batch size, n heads, query len, head dim]
        x = torch.matmul(self.dropout(attention), V)

        # [batch size, query len, n heads, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # [batch size, query len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)

        # [batch size, query len, hid dim]
        x = self.fc_o(x)

        return x, attention