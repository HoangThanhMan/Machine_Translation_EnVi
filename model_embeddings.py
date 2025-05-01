import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    def __init__(self, embed_size, vocab):
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        self.source = None
        self.target = None

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']
        
        self.source = nn.Embedding(len(vocab.src), embed_size, src_pad_token_idx)
        self.target = nn.Embedding(len(vocab.tgt), embed_size, tgt_pad_token_idx)
