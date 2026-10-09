# Transformer from Scratch (Vaswani et al., 2017)

A modular, pure PyTorch implementation of the full Sequence-to-Sequence (Seq2Seq) Transformer architecture as described in *"Attention Is All You Need"* (Vaswani et al., 2017).

## Features
- **Scaled Dot-Product Attention & Multi-Head Attention:** Scaled dot-product similarity with multi-head projections.
- **Sinusoidal Positional Encoding:** Invariant spatial context injected via sine and cosine waves.
- **Encoder & Decoder Stacks:** Full residual skip-connections and layer normalization (Post-LN).
- **Dynamic Masking Pipeline:** Source padding masks, target causal lower-triangular masks, and target padding masks.
- **Teacher-Forcing Training Engine:** Shifted target inputs, `CrossEntropyLoss(ignore_index=PAD_IDX)`, and inverse square-root warmup scheduler.
- **Inference Routines:** Autoregressive greedy decoding and beam search decoding.

## Project Structure
```text
transformer_project/
├── model.py        # MHA, Positional Encoding, Encoder, Decoder, Seq2Seq Wrapper
├── dataset.py      # Dynamic batch collation and 4D mask generation
├── train.py        # Training loop, scheduler, and loss handling
├── generate.py     # Greedy and Beam Search autoregressive decoding
└── requirements.txt



##Getting started
# 1. Clone repository
git clone <your-repo-url>
cd transformer_project

# 2. Setup environment
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 3. Run autoregressive inference check
python generate.py