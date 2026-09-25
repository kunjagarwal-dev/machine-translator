"""
PyTorch Dataset + collate_fn for the translation task.
Handles variable-length sequences with dynamic per-batch padding.
"""

import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, src_sentences: list[str], tgt_sentences: list[str],
                 src_vocab, tgt_vocab):
        assert len(src_sentences) == len(tgt_sentences), \
            "Source and target must have the same number of sentences"
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_ids = self.src_vocab.encode(self.src_sentences[idx])

        tgt_ids = self.tgt_vocab.encode(self.tgt_sentences[idx])
        tgt_ids = [self.tgt_vocab.START_IDX] + tgt_ids + [self.tgt_vocab.END_IDX]

        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_idx: int = 0):
    """
    Pads a batch of (src_ids, tgt_ids) pairs to the batch's own max length.
    Returns padded src/tgt tensors plus their original (unpadded) lengths.
    """
    src_batch, tgt_batch = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch], dtype=torch.long)

    src_max_len = int(max(src_lengths).item())
    tgt_max_len = int(max(tgt_lengths).item())

    src_padded = torch.full((len(src_batch), src_max_len), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(tgt_batch), tgt_max_len), pad_idx, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt

    return src_padded, tgt_padded, src_lengths, tgt_lengths