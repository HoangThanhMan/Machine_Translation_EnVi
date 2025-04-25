import yaml
from tqdm import tqdm

import torch
from torch.utils.data import random_split


def pad_tensor(sents, pad_id):
    # lengths = torch.tensor([len(s) for s in sents])
    padded_tensor = torch.nn.utils.rnn.pad_sequence(sents,
                                                    batch_first=True,
                                                    padding_value=pad_id)
    
    return padded_tensor

def get_config(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def read_sents(fpath, is_lowercase=False):
    sents = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            sents.append(line.rstrip("\n"))
        
    if is_lowercase:
        return [s.lower() for s in sents]
    else:
        return sents

def is_valid_sentence(tokenized_sent, max_len):
    return (len(tokenized_sent) < max_len and len(tokenized_sent) > 0)

def tokenize_and_remove_invalid_sents(src_sents, tgt_sents, max_len, src_tokenize, tgt_tokenize):
    src_tokenized_sents = []
    tgt_tokenized_sents = []

    total_lines = len(src_sents)
    for line in tqdm(range(total_lines), desc="Read lines"):
        src_tok = src_tokenize(src_sents[line])
        tgt_tok = tgt_tokenize(tgt_sents[line])
        if is_valid_sentence(src_tok, max_len) and is_valid_sentence(tgt_tok, max_len):
            src_tokenized_sents.append(src_tok)
            tgt_tokenized_sents.append(tgt_tok)

    print(f"Remove {total_lines - len(src_tokenized_sents)} invalid sentences")

    return src_tokenized_sents, tgt_tokenized_sents