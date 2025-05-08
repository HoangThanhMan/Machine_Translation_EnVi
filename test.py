import os
import argparse
import torch
from torch import nn

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Local modules
import utils.data_utils as data_utils
import utils.model_utils as model_utils
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.nmt import NMT
from models.tokenizer import EnTokenizer


def load_config(config_fpath):
    print(f"Loading config from {config_fpath}...")
    config = data_utils.get_config(config_fpath)
    return config


def load_tokenizers(checkpoint_dir, vocab_config):
    src_vocab_path = os.path.join(checkpoint_dir, vocab_config["src"])
    tgt_vocab_path = os.path.join(checkpoint_dir, vocab_config["tgt"])
    return EnTokenizer(src_vocab_path), EnTokenizer(tgt_vocab_path)


def load_dataloader(checkpoint_dir, dataloaders_file):
    dataloaders_path = os.path.join(checkpoint_dir, dataloaders_file)
    dataloaders = torch.load(dataloaders_path, weights_only=False)
    return dataloaders["test_loader"]


def build_model(config, src_tok, tgt_tok, device):
    encoder = Encoder(
        input_dim=len(src_tok.vocab),
        hid_dim=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        pf_dim=config["ffn_hidden"],
        dropout=config["drop_prob"],
        device=device,
        max_length=config["max_len"]
    )

    decoder = Decoder(
        input_dim=len(tgt_tok.vocab),
        hid_dim=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        pf_dim=config["ffn_hidden"],
        dropout=config["drop_prob"],
        device=device,
        max_len=config["max_len"]
    )

    return NMT(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=src_tok.vocab.pad_id,
        tgt_pad_idx=tgt_tok.vocab.pad_id,
        device=device
    ).to(device)


def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])


def main(config_fpath="config.yml"):
    config = load_config(config_fpath)
    checkpoint = config["checkpoint"]

    print("Loading data and tokenizers...")
    test_loader = load_dataloader(checkpoint["dir"], checkpoint["dataloaders"])
    src_tok, tgt_tok = load_tokenizers(checkpoint["dir"], checkpoint["vocab"])

    print("Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, src_tok, tgt_tok, device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.vocab.pad_id)

    best_ckpt_path = os.path.join(checkpoint["dir"], checkpoint["best"])
    load_checkpoint(model, best_ckpt_path)

    print(f"The model has {model_utils.count_parameters(model):,} trainable parameters")

    print("Starting evaluation...")
    model_utils.test(model, test_loader, criterion, src_tok, tgt_tok, config["max_len"])
    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the NMT model")
    parser.add_argument("--config", default="config.yml", help="Path to config file", dest="config_fpath")
    args = parser.parse_args()

    main(args.config_fpath)
