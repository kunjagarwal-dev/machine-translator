"""
Manual PyTorch training loop utilities for the GRU + Attention Seq2Seq model.
Differs from train.py (LSTM baseline): encoder returns (outputs, hidden)
instead of (hidden, cell), and the decoder needs encoder_outputs + src_mask
at every step for attention.
"""

import torch


def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0.0

    for i, (src, tgt, src_lens, tgt_lens) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        outputs = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        # outputs: (batch, tgt_len, vocab_size)

        output_dim = outputs.shape[-1]
        outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim)
        targets_flat = tgt[:, 1:].reshape(-1)

        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if i % 50 == 0:
            print(f"  Batch {i}/{len(loader)} — loss: {loss.item():.4f}")

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
def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_len=80, return_attention=False):
    """Greedy decoding: translate a single raw sentence string."""
    model.eval()

    src_ids = src_vocab.encode(sentence)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, src_len)

    encoder_outputs, hidden = model.encoder(src_tensor)
    src_mask = model.create_src_mask(src_tensor)

    input_token = torch.tensor([tgt_vocab.START_IDX], dtype=torch.long, device=device)
    output_ids = []
    attn_weights_list = [] if return_attention else None
    attn_weights_list = [] if return_attention else None

    for _ in range(max_len):
        logits, hidden, attn_weights = model.decoder(input_token, hidden, encoder_outputs, src_mask)
        top1 = logits.argmax(1)
        token_id = top1.item()

        if return_attention:
            assert attn_weights_list is not None
            attn_weights_list.append(attn_weights.squeeze(0).cpu())

        if token_id == tgt_vocab.END_IDX:
            break

        output_ids.append(token_id)
        input_token = top1

    translation = tgt_vocab.decode(output_ids)

    if return_attention:
        return translation, attn_weights_list
    return translation