from itertools import chain
from collections import Counter
import torch
from utils.data_utils import pad_tensor

class Vocabulary:
    def __init__(self):
        self.special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.word2id = {token : idx for idx, token in enumerate(self.special_tokens)}
        self.id2word = {idx : token for token, idx in self.word2id.items()}

        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.unk_id = self.word2id["<unk>"]
        self.pad_id = self.word2id["<pad>"]
        self.bos_id = self.word2id["<bos>"]
        self.eos_id = self.word2id["<eos>"]

    def __len__(self):
        return len(self.word2id)
    
    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)
    
    def __contains__(self, word):
        return word in self.word2id

    def add(self, word):
        if word not in self:
            word_id = len(self.word2id)
            self.word2id[word] = word_id
            self.id2word[word_id] = word
            return word_id
        
        return self[word]

    def words2tensor(self, words, add_bos_eos=False):
        if add_bos_eos:
            words = ["<bos>"] + words + ["<eos>"]

        ids = [self[word] for word in words]
        return torch.tensor(ids, dtype=torch.int64)

    def sents2tensors(self, tokenized_sents, add_bos_eos=False):
        return [self.words2tensor(s, add_bos_eos) for s in tokenized_sents]
    
    def tensor2words(self, tensor):
        return [self.id2word[idx.item()] for idx in tensor]

    def tensors2sents(self, tensors):
        return [self.tensor2words(tensor) for tensor in tensors]

    def add_words(self, tokenized_sents, min_freq=1, vocab_size=None):
        word_freq = Counter(chain(*tokenized_sents))
        frequent_words = [w for w, freq in word_freq.items() if freq >= min_freq]

        print(f"Total number of tokens in the corpus: {len(word_freq)}")
        print(f"Number of tokens appearing >= {min_freq} times: {len(frequent_words)}")

        if vocab_size is not None:
            frequent_words = sorted(frequent_words, key=lambda w: word_freq[w], reverse=True)[:vocab_size]

        print(f"Total number of Vocabulary tokens (excluding special tokens): {len(frequent_words)}")
        for word in frequent_words:
            self.add(word)

class ParallelVocabulary:
    def __init__(self, src_vocab, tgt_vocab, is_sorted, device):
        self.src = src_vocab
        self.tgt = tgt_vocab
        self.is_sorted = is_sorted
        self.device = device

    def collate_fn(self, examples):
        src_sents = [pair["src"] for pair in examples]
        tgt_sents = [pair["tgt"] for pair in examples]
        if self.is_sorted:
            pairs = zip(src_sents, tgt_sents)
            sorted_pairs = zip(*sorted(pairs, key= lambda x: (len(x[0]), len(x[1])), reverse=True))
            src_sents, tgt_sents = tuple(list(sorted_sents) for sorted_sents in sorted_pairs)
        return {"src": pad_tensor(src_sents, self.src.pad_id).to(self.device),
                "tgt": pad_tensor(tgt_sents, self.tgt.pad_id).to(self.device)}