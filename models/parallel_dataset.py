from torch.utils.data import Dataset

class ParallelDataset(Dataset):
    def __init__(self, src_tokenized_sents, tgt_tokenized_sents,
                 src_tokenizer, tgt_tokenizer, is_sorted=True):
        if is_sorted:
            pair_tokenized_sents = zip(src_tokenized_sents, tgt_tokenized_sents)
            sorted_pair = sorted(pair_tokenized_sents,key=lambda pair: len(pair[0]))
            self.src_tokenized_sents, self.tgt_tokenized_sents = list(zip(*sorted_pair))
        else:
            self.src_tokenized_sents = src_tokenized_sents
            self.tgt_tokenized_sents = tgt_tokenized_sents

        self.src_mapped_sents = src_tokenizer.vocab.sents2tensors(self.src_tokenized_sents, add_bos_eos=True)
        self.tgt_mapped_sents = tgt_tokenizer.vocab.sents2tensors(self.tgt_tokenized_sents, add_bos_eos=True)

    def __len__(self):
        return len(self.src_mapped_sents)

    def __getitem__(self, index):
        if isinstance(index, int):
            return {
                'src': self.src_mapped_sents[index],
                'tgt': self.tgt_mapped_sents[index]
            }
        elif isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop, index.step)]
        elif isinstance(index, list):
            return [self[i] for i in index]
        else:
            raise TypeError("Invalid argument type.")