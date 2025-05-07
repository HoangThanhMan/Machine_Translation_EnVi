import torch
from models.transformer.encoder import Encoder

# Thiết lập tham số
BATCH_SIZE = 2
SRC_LEN = 10
INPUT_DIM = 100  # số lượng token trong vocab
HID_DIM = 64
N_LAYERS = 2
N_HEADS = 8
PF_DIM = 256
DROPOUT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tạo encoder
encoder = Encoder(input_dim=INPUT_DIM,
                  hid_dim=HID_DIM,
                  n_layers=N_LAYERS,
                  n_heads=N_HEADS,
                  pf_dim=PF_DIM,
                  dropout=DROPOUT,
                  device=DEVICE).to(DEVICE)

# Dữ liệu đầu vào ngẫu nhiên (batch of sequences)
src = torch.randint(0, INPUT_DIM, (BATCH_SIZE, SRC_LEN)).to(DEVICE)  # [batch size, src len]

# Mặt nạ nguồn (cho attention, giả sử không mask gì cả)
src_mask = torch.ones((BATCH_SIZE, 1, 1, SRC_LEN)).to(DEVICE)  # [batch size, 1, 1, src len]

# Forward qua encoder
with torch.no_grad():
    output = encoder(src, src_mask)

print("Input shape:", src.shape)         # [batch size, src len]
print("Output shape:", output.shape)     # [batch size, src len, hid dim]
