import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import argparse

import torch
from torch.utils.data import DataLoader

import utils.data_utils as data_utils
import utils.other_utils as other_utils
from models.parallel_dataset import ParallelDataset
from models.tokenizer import EnTokenizer, ViTokenizer
from models.vocabulary import ParallelVocabulary

def load_dataloader_from_fpath(pair_fpath, src_tok, tgt_tok, batch_size, max_len,
                               device, is_lowercase, is_train=False, min_freq=1, 
                               vocab_size=None):
    
    src_sents = data_utils.read_sents(pair_fpath["src"], is_lowercase=True)
    tgt_sents = data_utils.read_sents(pair_fpath["tgt"], is_lowercase=True)

    src_tokenized_sents, tgt_tokenized_sents = data_utils.tokenize_and_remove_invalid_sents(src_sents, tgt_sents, 
                                                                                            max_len, 
                                                                                            src_tok.tokenize, 
                                                                                            tgt_tok.tokenize)
    
    if is_train:
        src_tok.built_vocab(src_sents, is_tokenized=True, min_freq=min_freq, vocab_size=vocab_size)
        tgt_tok.built_vocab(tgt_sents, is_tokenized=False, min_freq=min_freq, vocab_size=vocab_size)

    dataset = ParallelDataset(src_tokenized_sents, tgt_tokenized_sents, src_tok, tgt_tok)
    parallel_vocab = ParallelVocabulary(src_tok.vocab, tgt_tok.vocab, is_sorted=False, device=device)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=parallel_vocab.collate_fn,
                        shuffle=is_train)

    if is_train:
        return loader, src_tok, tgt_tok
    else:
        return loader


def main(config_fpath="config.yml"):
    config = data_utils.get_config(config_fpath)
    
    for key, val in config.items():
        globals()[key] = val

    src_tokenizer = EnTokenizer()
    tgt_tokenizer = ViTokenizer()

    print("Load tokenizers...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

