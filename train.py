import os
import math
import argparse
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

import utils.data_utils as data_utils
import utils.model_utils as model_utils
import utils.other_utils as other_utils
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.nmt import NMT
from models.tokenizer import EnTokenizer, ViTokenizer

def load_config_and_globals(config_fpath):
    global config, checkpoint, d_model, n_layers, n_heads, ffn_hidden, drop_prob, max_len, clip, init_lr, total_epoch
    print(f"Load config file {config_fpath}...")
    config = data_utils.get_config(config_fpath)
    checkpoint = config["checkpoint"]
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    ffn_hidden = config["ffn_hidden"]
    drop_prob = config["drop_prob"]
    max_len = config["max_len"]
    clip = config["clip"]
    init_lr = config["init_lr"]
    total_epoch = config["total_epoch"]
    

def load_dataloaders_and_tokenizers():
    print("Load prepared dataloaders & tokenizers...")
    dataloaders_fpath = os.path.join(checkpoint["dir"], checkpoint["dataloaders"])
    dataloaders = torch.load(dataloaders_fpath, weights_only=False)

    train_loader = dataloaders["train_loader"]
    valid_loader = dataloaders["valid_loader"]

    src_vocab_fpath = os.path.join(checkpoint["dir"], checkpoint["vocab"]["src"])
    tgt_vocab_fpath = os.path.join(checkpoint["dir"], checkpoint["vocab"]["tgt"])

    src_tok = EnTokenizer(src_vocab_fpath)
    tgt_tok = ViTokenizer(tgt_vocab_fpath)

    return train_loader, valid_loader, src_tok, tgt_tok

def build_model(src_tok, tgt_tok, device):
    print("Build model & optimizer & criterion...")

    enc = Encoder(input_dim=len(src_tok.vocab), hid_dim=d_model, n_layers=n_layers,
                  n_heads=n_heads, pf_dim=ffn_hidden, dropout=drop_prob,
                  device=device, max_length=max_len)

    dec = Decoder(output_dim=len(tgt_tok.vocab), hid_dim=d_model, n_layers=n_layers,
                  n_heads=n_heads, pf_dim=ffn_hidden, dropout=drop_prob,
                  device=device, max_length=max_len)

    model = NMT(enc, dec, src_tok.vocab.pad_id, tgt_tok.vocab.pad_id, device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.vocab.pad_id)

    return model, criterion

def load_checkpoint(model):
    print("Load checkpoint...")
    begin_epoch = 1
    best_loss = float("inf")

    best_checkpoint_fpath = os.path.join(checkpoint["dir"], checkpoint["best"])
    last_checkpoint_fpath = os.path.join(checkpoint["dir"], checkpoint["last"])

    optimizer = Adam(params=model.parameters(), lr=init_lr)

    if other_utils.exist(last_checkpoint_fpath):
        checkpoint_dict = torch.load(last_checkpoint_fpath)
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        best_loss = checkpoint_dict["loss"]
        begin_epoch = checkpoint_dict["epoch"] + 1
        if begin_epoch <= total_epoch:
            print(f"Continue from the last epoch {begin_epoch}...")
    else:
        print("Last checkpoint not found, initializing new model...")
        model.apply(model_utils.initialize_weights)

    return model, optimizer, begin_epoch, best_loss, best_checkpoint_fpath, last_checkpoint_fpath

def train_and_evaluate(model, train_loader, valid_loader, optimizer, criterion, 
                       src_tok, tgt_tok, device, 
                       begin_epoch, best_loss,
                       best_checkpoint_fpath, last_checkpoint_fpath):

    columns = ["epoch", "train_loss", "valid_loss", "valid_BLEU"]
    results_fpath = os.path.join(checkpoint["dir"], checkpoint["results"])
    results_df = pd.read_csv(results_fpath) if other_utils.exist(results_fpath) else pd.DataFrame(columns=columns)

    print("Start training & evaluating...")
    for epoch in range(begin_epoch, total_epoch+1):
        with other_utils.TimeContextManager() as timer:
            train_loss = model_utils.train(epoch, model, train_loader, optimizer, criterion, clip)
            valid_loss = model_utils.evaluate(model, valid_loader, criterion)
            valid_BLEU = model_utils.calculate_dataloader_bleu(
                valid_loader, src_tok, tgt_tok, model, device, max_len=max_len, teacher_forcing=True) * 100

        epoch_mins, epoch_secs = timer.get_time()

        results_df.loc[len(results_df)] = epoch, train_loss, valid_loss, valid_BLEU
        results_df.to_csv(results_fpath, index=False)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({"epoch": epoch, "loss": best_loss,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, best_checkpoint_fpath)

        torch.save({"epoch": epoch, "loss": valid_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, last_checkpoint_fpath)

        print(f"Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}")
        print(f"\tBLEU Score: {valid_BLEU:.3f}")

    print("Finish training!")

def main(config_fpath="config.yml"):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Stop printing tensorflow's logs

    load_config_and_globals(config_fpath)

    train_loader, valid_loader, src_tok, tgt_tok = load_dataloaders_and_tokenizers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion = build_model(src_tok, tgt_tok, device)

    model, optimizer, begin_epoch, best_loss, best_ckpt, last_ckpt = load_checkpoint(model)

    print(f"The model has {model_utils.count_parameters(model):,} trainable parameters")

    train_and_evaluate(model, train_loader, valid_loader, optimizer, criterion,
                       src_tok, tgt_tok, device,
                       begin_epoch, best_loss, best_ckpt, last_ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate the NMT model")
    parser.add_argument("--config", default="config.yml", help="path to config file", dest="config_fpath")
    args = parser.parse_args()
    main(**vars(args))