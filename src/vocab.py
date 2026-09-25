"""
Vocabulary class for building word-to-index mappings for source/target languages.
Build separately for English and Hindi, using only the training split.
"""

from collections import Counter


class Vocab:
    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"

    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3

    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.START_TOKEN: self.START_IDX,
            self.END_TOKEN: self.END_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def build(self, sentences: list[str]):
        """Build vocab from a list of raw sentences (whitespace tokenized)."""
        counter = Counter()
        for sentence in sentences:
            tokens = sentence.strip().split()
            counter.update(tokens)

        next_idx = len(self.word2idx)
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = next_idx
                self.idx2word[next_idx] = word
                next_idx += 1

        print(f"Vocab built: {len(self.word2idx)} tokens "
              f"(min_freq={self.min_freq}, raw unique tokens={len(counter)})")

    def encode(self, sentence: str) -> list[int]:
        """Convert a sentence string into a list of token ids (no START/END added here)."""
        tokens = sentence.strip().split()
        return [self.word2idx.get(tok, self.UNK_IDX) for tok in tokens]

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        """Convert a list of token ids back into a sentence string."""
        words = []
        for idx in ids:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if strip_special and word in (
                self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN
            ):
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)