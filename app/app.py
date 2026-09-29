"""
EN <-> HI Machine Translation demo.
Compares LSTM, GRU+Attention, MarianMT (zero-shot), and MarianMT (fine-tuned).

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import os
import pickle
import sys

import streamlit as st
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.models.lstm_se12seq import Encoder as LSTMEncoder, Decoder as LSTMDecoder, Seq2Seq as LSTMSeq2Seq
from src.models.gru_attention import Encoder as GRUEncoder, Decoder as GRUDecoder, Seq2Seq as GRUSeq2Seq
from src.train import translate_sentence as lstm_translate
from src.train_attention import translate_sentence as gru_translate

st.set_page_config(page_title="EN / HI Translator", page_icon="~", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 128
HIDDEN_DIM = 256

# ---------- styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:ital,wght@0,500;0,600;1,500&display=swap');

:root {
    --bg: #1A1B2E;
    --bg-panel: #232544;
    --text: #F2EFE9;
    --text-dim: #9C9CB5;
    --accent: #E8A33D;
    --accent-dim: #6B5730;
    --border: #33355A;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }

.app-title {
    font-family: 'Lora', serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.1rem;
}
.app-subtitle {
    color: var(--text-dim);
    font-size: 0.95rem;
    margin-bottom: 1.8rem;
}

div[data-testid="stRadio"] > div { flex-direction: row; gap: 0.5rem; }
div[data-testid="stRadio"] label {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 1rem;
}

textarea {
    background: var(--bg-panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
}

.output-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    min-height: 140px;
    font-family: 'Lora', serif;
    font-size: 1.15rem;
    line-height: 1.6;
    color: var(--text);
}

.model-label {
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 0.4rem;
    letter-spacing: 0.02em;
}

.bleu-tag {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border-radius: 6px;
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
    font-family: 'Inter', sans-serif;
    margin-left: 0.5rem;
}

button[data-baseweb="tab"] { color: var(--text-dim) !important; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

.stButton button {
    background: var(--accent);
    color: #1A1B2E;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.6rem;
}
.stButton button:hover { background: #f0b45c; color: #1A1B2E; }
</style>
""", unsafe_allow_html=True)

# ---------- model registry ----------
MODEL_INFO = {
    "LSTM (no attention)": {"bleu": 0.09, "type": "custom"},
    "GRU + Attention": {"bleu": 0.61, "type": "custom"},
    "MarianMT (zero-shot)": {"bleu": {"en_hi": 9.85, "hi_en": 13.42}, "type": "marian_base"},
    "MarianMT (fine-tuned)": {"bleu": {"en_hi": 13.00, "hi_en": 14.98}, "type": "marian_ft"},
}


# ---------- loaders (cached so weights load once per session) ----------
@st.cache_resource
def load_vocabs():
    with open("data/processed/en_vocab.pkl", "rb") as f:
        en_vocab = pickle.load(f)
    with open("data/processed/hi_vocab.pkl", "rb") as f:
        hi_vocab = pickle.load(f)
    return en_vocab, hi_vocab


@st.cache_resource
def load_lstm_model():
    en_vocab, hi_vocab = load_vocabs()
    encoder = LSTMEncoder(len(en_vocab), EMBED_DIM, HIDDEN_DIM, pad_idx=en_vocab.PAD_IDX)
    decoder = LSTMDecoder(len(hi_vocab), EMBED_DIM, HIDDEN_DIM, pad_idx=hi_vocab.PAD_IDX)
    model = LSTMSeq2Seq(encoder, decoder, DEVICE, start_idx=hi_vocab.START_IDX, end_idx=hi_vocab.END_IDX).to(DEVICE)
    model.load_state_dict(torch.load("models/lstm_baseline/model.pt", map_location=DEVICE))
    model.eval()
    return model, en_vocab, hi_vocab


@st.cache_resource
def load_gru_attention_model():
    en_vocab, hi_vocab = load_vocabs()
    encoder = GRUEncoder(len(en_vocab), EMBED_DIM, HIDDEN_DIM, pad_idx=en_vocab.PAD_IDX)
    decoder = GRUDecoder(len(hi_vocab), EMBED_DIM, HIDDEN_DIM, pad_idx=hi_vocab.PAD_IDX)
    model = GRUSeq2Seq(encoder, decoder, DEVICE, start_idx=hi_vocab.START_IDX,
                        end_idx=hi_vocab.END_IDX, pad_idx=en_vocab.PAD_IDX).to(DEVICE)
    model.load_state_dict(torch.load("models/gru_attention/model.pt", map_location=DEVICE))
    model.eval()
    return model, en_vocab, hi_vocab


@st.cache_resource
def load_marian_base(direction):
    from transformers import MarianMTModel, MarianTokenizer
    name = "Helsinki-NLP/opus-mt-en-hi" if direction == "en_hi" else "Helsinki-NLP/opus-mt-hi-en"
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name).to(DEVICE)
    return tok, model


@st.cache_resource
def load_marian_finetuned(direction):
    from transformers import MarianMTModel, MarianTokenizer
    path = f"models/marian_{direction}_finetuned"
    tok = MarianTokenizer.from_pretrained(path)
    model = MarianMTModel.from_pretrained(path).to(DEVICE)
    return tok, model


def translate_marian(text, tok, model):
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = model.generate(**inputs, max_length=80, num_beams=4, early_stopping=True)
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def translate_custom(text, direction, model_key):
    # Both custom (from-scratch) models were only trained EN -> HI in this project.
    if direction != "en_hi":
        return "This model was only trained for English -> Hindi in this project."

    try:
        if model_key == "LSTM (no attention)":
            model, en_vocab, hi_vocab = load_lstm_model()
            return lstm_translate(model, text, en_vocab, hi_vocab, DEVICE)
        else:
            model, en_vocab, hi_vocab = load_gru_attention_model()
            return gru_translate(model, text, en_vocab, hi_vocab, DEVICE)
    except FileNotFoundError:
        return "[checkpoint not found - place model.pt under models/ to enable this tab]"


# ---------- header ----------
st.markdown('<div class="app-title">English &harr; Hindi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Four translation systems, side by side &mdash; '
    'from a plain LSTM up to a fine-tuned MarianMT.</div>',
    unsafe_allow_html=True
)

# ---------- direction + input ----------
col_dir, _ = st.columns([1, 3])
with col_dir:
    direction_label = st.radio("Direction", ["English -> Hindi", "Hindi -> English"], label_visibility="collapsed")

direction = "en_hi" if direction_label == "English -> Hindi" else "hi_en"
src_lang = "English" if direction == "en_hi" else "Hindi"

source_text = st.text_area(f"{src_lang} text", height=110, placeholder=f"Type {src_lang} here...")

translate_clicked = st.button("Translate")

st.write("")

# ---------- model tabs ----------
tab_labels = list(MODEL_INFO.keys())
tabs = st.tabs(tab_labels)

for tab, model_name in zip(tabs, tab_labels):
    with tab:
        info = MODEL_INFO[model_name]
        bleu = info["bleu"][direction] if isinstance(info["bleu"], dict) else info["bleu"]

        st.markdown(
            f'<div class="model-label">{model_name.upper()}'
            f'<span class="bleu-tag">BLEU {bleu:.2f}</span></div>',
            unsafe_allow_html=True
        )

        if translate_clicked and source_text.strip():
            with st.spinner("Translating..."):
                if info["type"] == "custom":
                    result = translate_custom(source_text, direction, model_name)
                else:
                    loader = load_marian_finetuned if info["type"] == "marian_ft" else load_marian_base
                    tok, model = loader(direction)
                    result = translate_marian(source_text, tok, model)
            st.markdown(f'<div class="output-card">{result}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="output-card" style="color: var(--text-dim);">'
                'Translation will appear here.</div>',
                unsafe_allow_html=True
            )

st.write("")
st.markdown(
    '<div style="color: var(--text-dim); font-size: 0.85rem; border-top: 1px solid var(--border); '
    'padding-top: 1rem; margin-top: 1rem;">'
    'Trained on a subset of the IIT Bombay English-Hindi parallel corpus. '
    'BLEU scores computed on a held-out 10k-sentence test split.'
    '</div>',
    unsafe_allow_html=True
)