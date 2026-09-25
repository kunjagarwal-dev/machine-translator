"""
LSTM Seq2Seq baseline (no attention).
Encoder compresses source sentence into a fixed context; Decoder generates
target tokens one step at a time from that context.
"""

import random
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.embedding(src)  # (batch, src_len, embed_dim)
        _, (hidden, cell) = self.lstm(embedded)
        # hidden, cell: (num_layers, batch, hidden_dim)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, cell):
        # input_token: (batch,) -- single token per sequence at this timestep
        input_token = input_token.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(input_token)  # (batch, 1, embed_dim)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: (batch, 1, hidden_dim)
        logits = self.fc_out(output.squeeze(1))  # (batch, vocab_size)
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, start_idx, end_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (batch, src_len), tgt: (batch, tgt_len) -- tgt includes <START>...<END>
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=self.device)

        hidden, cell = self.encoder(src)

        # first decoder input is <START>, which is tgt[:, 0]
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            logits, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = logits

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs  # (batch, tgt_len, tgt_vocab_size)