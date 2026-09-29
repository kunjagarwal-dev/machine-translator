# Machine Translation: English &harr; Hindi

Four translation systems built and compared on the same English-Hindi parallel corpus, from a plain LSTM up to a fine-tuned MarianMT model.

## Results

| Model                       | Direction      | BLEU      |
| --------------------------- | -------------- | --------- |
| LSTM Seq2Seq (no attention) | EN&rarr;HI     | 0.09      |
| GRU + Attention             | EN&rarr;HI     | 0.61      |
| MarianMT (zero-shot)        | EN&rarr;HI     | 9.85      |
| **MarianMT (fine-tuned)**   | **EN&rarr;HI** | **13.00** |
| MarianMT (zero-shot)        | HI&rarr;EN     | 13.42     |
| **MarianMT (fine-tuned)**   | **HI&rarr;EN** | **14.98** |

BLEU computed with `sacrebleu` on a held-out 10,000-sentence test split, identical across all models.

## What this project shows

- A from-scratch LSTM encoder-decoder (no attention) as a weak baseline
- A from-scratch GRU with Bahdanau attention, showing the concrete improvement attention gives over a fixed-context bottleneck
- MarianMT (Helsinki-NLP `opus-mt`) evaluated zero-shot, as a strong pretrained reference point
- MarianMT fine-tuned with a **manual PyTorch training loop** (no HuggingFace `Trainer`) on this project's own data, showing the gain from fine-tuning a pretrained model over using it untouched

All four models are evaluated on the exact same test split for a fair comparison.

## Dataset

[IIT Bombay English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi), filtered and deduplicated:

- Dropped exact-duplicate and empty pairs
- Length-filtered to &le;64 English words / &le;80 Hindi words (99th-percentile based cutoff)
- Trained on a random 180,000-pair subset (seed 42) for tractable training time on consumer/free-tier GPU hardware; evaluated on a fixed 10,000-pair test subset

## Architecture notes

**LSTM baseline** &mdash; `Embedding` &rarr; `LSTM` encoder compresses the source sentence into a single fixed-size context vector; the decoder generates the target sequence from that one vector alone. This bottleneck is the main limitation the next model addresses.

**GRU + Attention** &mdash; the encoder keeps its output at every timestep instead of only the final one. At each decoding step, a Bahdanau-style attention module scores every source position against the decoder's current state and builds a fresh, situation-specific context vector. Roughly a 7x BLEU improvement over the no-attention baseline at the same training budget.

**MarianMT fine-tuning** &mdash; full fine-tuning (not LoRA/QLoRA &mdash; unnecessary at this model size) via a hand-written training loop: tokenize with `text_target=`, mask padding positions in `labels` with `-100` so they don't contribute to the loss, forward pass with `labels` included so the model returns `loss` directly, backward, step.

## Known limitations

- Both from-scratch models were trained on a **180K-pair subset**, not the full ~1.4M-pair filtered corpus, and only for a few epochs, due to training-time constraints on consumer GPU hardware (RTX 4050) and free-tier cloud GPUs (Colab T4, Kaggle T4). BLEU scores for these two models reflect that limited training budget, not a ceiling on the architecture itself.
- The from-scratch LSTM and GRU+Attention models were trained **EN&rarr;HI only**; MarianMT was fine-tuned in both directions.
- `embed_dim=128, hidden_dim=128` were used for both from-scratch models, chosen for GPU memory constraints (an earlier attempt at `hidden_dim=512` triggered out-of-memory errors) and kept identical across LSTM and GRU+Attention for a fair, capacity-matched comparison.
- MarianMT was fine-tuned for 2 epochs per direction; both train and validation loss were still decreasing at that point, so further training would likely improve results further.

## Project structure

```
mt-en-hi/
├── data/               # raw + processed/filtered corpus (gitignored)
├── notebooks/          # one notebook per day of the project
├── src/                # vocab, dataset, model architectures, training loops
├── models/             # saved checkpoints (gitignored)
├── results/            # BLEU scores + sample translations, per model
├── app/                # Streamlit demo
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the demo

```bash
streamlit run app/streamlit_app.py
```

Compares all four models side by side, in both translation directions.

## Tech stack

PyTorch, HuggingFace Transformers & Datasets, sacrebleu, Streamlit. Trained locally (RTX 4050) and on Google Colab / Kaggle (T4 GPUs) as needed for compute.
