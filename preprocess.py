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
    
    src_sents = data_utils.read_sents(pair_fpath["src"], is_lowercase)
    tgt_sents = data_utils.read_sents(pair_fpath["tgt"], is_lowercase)

    src_tokenized_sents, tgt_tokenized_sents = data_utils.tokenize_and_remove_invalid_sents(src_sents, tgt_sents, 
                                                                                            max_len, 
                                                                                            src_tok.tokenize, 
                                                                                            tgt_tok.tokenize)
    
    if is_train:
        print("Build vocab...")
        src_tok.build_vocab(src_tokenized_sents, is_tokenized=True, min_freq=min_freq, vocab_size=vocab_size)
        tgt_tok.build_vocab(tgt_tokenized_sents, is_tokenized=True, min_freq=min_freq, vocab_size=vocab_size)

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
    # Load config
    config = data_utils.get_config(config_fpath)
    path = config["path"]
    checkpoint = config["checkpoint"]

    src_tokenizer = EnTokenizer()
    tgt_tokenizer = EnTokenizer()

    print("Load DataLoader...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load train
    print("Load training set")
    train_loader, src_tokenizer, tgt_tokenizer = load_dataloader_from_fpath(path["train"],
                                                                                src_tokenizer, 
                                                                                tgt_tokenizer, 
                                                                                config["batch_size"], 
                                                                                config["max_len"], device, 
                                                                                is_lowercase=True, 
                                                                                is_train=True,
                                                                                min_freq=config["min_freq"], 
                                                                                vocab_size=config["vocab_size"])

    # Load valid
    print("Load validation set")
    valid_loader = load_dataloader_from_fpath(path["valid"],
                                                src_tokenizer,
                                                tgt_tokenizer,
                                                config["batch_size"],
                                                config["max_len"],
                                                device,
                                                is_lowercase=True,
                                                is_train=False)
    
    # Load test
    print("Load test set")
    test_loader = load_dataloader_from_fpath(path["test"],
                                                src_tokenizer,
                                                tgt_tokenizer,
                                                config["batch_size"],
                                                config["max_len"],
                                                device,
                                                is_lowercase=True,
                                                is_train=False)
    
    data_loaders = {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
    }

    print("Load file paths and save ...")

    other_utils.create_dir(checkpoint["dir"])
    src_tokenizer.save_vocab(os.path.join(checkpoint["dir"], checkpoint["vocab"]["src"]))
    tgt_tokenizer.save_vocab(os.path.join(checkpoint["dir"], checkpoint["vocab"]["tgt"]))
    torch.save(data_loaders, os.path.join(checkpoint["dir"], checkpoint["dataloaders"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess parallel datasets and train tokenizers")

    parser.add_argument("--config", default="config.yml", help="path to config file", dest="config_fpath")
    
    args = parser.parse_args()

    main(**vars(args))

