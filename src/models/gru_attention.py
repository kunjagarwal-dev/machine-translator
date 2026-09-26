"""
GRU + Bahdanau Attention Seq2Seq model.
Unlike the LSTM baseline, the decoder attends over ALL encoder outputs
at every timestep, instead of relying on a single fixed context vector.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.embedding(src)                    # (batch, src_len, embed_dim)
        outputs, hidden = self.gru(embedded)               # outputs: (batch, src_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim) -- keep for decoder init
        return outputs, hidden


class Attention(nn.Module):
    """Bahdanau-style additive attention."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        # decoder_hidden: (batch, hidden_dim)      -- current decoder step's hidden state
        # encoder_outputs: (batch, src_len, hidden_dim)
        src_len = encoder_outputs.shape[1]

        # repeat decoder_hidden across src_len so it can be compared to every encoder output
        decoder_hidden_rep = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, hidden_dim)

        # combine and score each source position
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden_rep, encoder_outputs), dim=2)))
        # energy: (batch, src_len, hidden_dim)
        scores = self.v(energy).squeeze(2)  # (batch, src_len)

        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=1)  # (batch, src_len), sums to 1 across src_len

        # weighted sum of encoder outputs -> context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden_dim)
        context = context.squeeze(1)  # (batch, hidden_dim)

        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = Attention(hidden_dim)
        # GRU input = embedding + context vector concatenated
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2 + embed_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, src_mask=None):
        # input_token: (batch,) -- single token per sequence at this timestep
        # hidden: (num_layers, batch, hidden_dim) -- previous decoder hidden state
        input_token = input_token.unsqueeze(1)          # (batch, 1)
        embedded = self.embedding(input_token)           # (batch, 1, embed_dim)

        # use top layer's hidden state to compute attention
        top_hidden = hidden[-1]                           # (batch, hidden_dim)
        context, attn_weights = self.attention(top_hidden, encoder_outputs, src_mask)
        context_unsq = context.unsqueeze(1)               # (batch, 1, hidden_dim)

        gru_input = torch.cat((embedded, context_unsq), dim=2)  # (batch, 1, embed_dim + hidden_dim)
        output, hidden = self.gru(gru_input, hidden)
        # output: (batch, 1, hidden_dim)

        output = output.squeeze(1)      # (batch, hidden_dim)
        embedded = embedded.squeeze(1)  # (batch, embed_dim)

        # combine output, context, and embedding before final projection (common trick, richer signal)
        logits = self.fc_out(torch.cat((output, context, embedded), dim=1))  # (batch, vocab_size)

        return logits, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, start_idx, end_idx, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx

    def create_src_mask(self, src):
        # (batch, src_len) -- 1 for real tokens, 0 for padding
        return (src != self.pad_idx).long()

    def forward(self, src, tgt, teacher_forcing_ratio=0.5, return_attention=False):
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=self.device)
        all_attn_weights = [] if return_attention else None

        encoder_outputs, hidden = self.encoder(src)
        src_mask = self.create_src_mask(src)

        input_token = tgt[:, 0]  # <START>

        for t in range(1, tgt_len):
            logits, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs, src_mask)
            outputs[:, t, :] = logits

            if return_attention:
                assert all_attn_weights is not None
                all_attn_weights.append(attn_weights.unsqueeze(1))  # (batch, 1, src_len)

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        if return_attention:
            assert all_attn_weights is not None
            all_attn_weights = torch.cat(all_attn_weights, dim=1)  # (batch, tgt_len-1, src_len)
            return outputs, all_attn_weights

        return outputs