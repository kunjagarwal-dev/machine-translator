"""
Manual PyTorch training loop utilities, reused across LSTM, GRU+Attention,
and MarianMT fine-tuning stages.
"""

import torch


def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0.0

    for src, tgt, src_lens, tgt_lens in loader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        outputs = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        # outputs: (batch, tgt_len, vocab_size) -- compare outputs[1:] to tgt[1:] (skip <START>)

        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:, :].reshape(-1, output_dim)
        targets = tgt[:, 1:].reshape(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for src, tgt, src_lens, tgt_lens in loader:
        src, tgt = src.to(device), tgt.to(device)

        outputs = model(src, tgt, teacher_forcing_ratio=0.0)  # no teacher forcing at eval

        output_dim = outputs.shape[-1]
        outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim)
        targets_flat = tgt[:, 1:].reshape(-1)

        loss = criterion(outputs_flat, targets_flat)
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_len=80):
    """Greedy decoding: translate a single raw sentence string."""
    model.eval()

    src_ids = src_vocab.encode(sentence)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, src_len)

    hidden, cell = model.encoder(src_tensor)

    input_token = torch.tensor([tgt_vocab.START_IDX], dtype=torch.long, device=device)
    output_ids = []

    for _ in range(max_len):
        logits, hidden, cell = model.decoder(input_token, hidden, cell)
        top1 = logits.argmax(1)
        token_id = top1.item()

        if token_id == tgt_vocab.END_IDX:
            break

        output_ids.append(token_id)
        input_token = top1

    return tgt_vocab.decode(output_ids)