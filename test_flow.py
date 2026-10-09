import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# 1. Positional Encoding Module
# ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        i = torch.arange(0, d_model, 2).float()
        div = 10000.0 ** (i / d_model)

        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        self.pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

    def forward(self, x):
        T = x.shape[1]
        return x + self.pe[:, :T].to(x.device)

# ─────────────────────────────────────────────────────────────
# 2. Multi-Head Attention Module
# ─────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, verbose=False):
        B, T, C = x.shape

        # Step 1: Linear projections + split into heads
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        if verbose:
            print(f"      [MHA] Q, K, V projected & transposed shape: {list(q.shape)}")

        # Step 2: Affinity dot-product scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if verbose:
            print(f"      [MHA] Pairwise similarity grid shape:      {list(scores.shape)}")

        # Step 3: Probability weights
        weights = F.softmax(scores, dim=-1)

        # Step 4: Context vector construction
        out = torch.matmul(weights, v)
        if verbose:
            print(f"      [MHA] Context heads output shape:          {list(out.shape)}")

        # Step 5: Merge heads & project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.w_o(out)
        if verbose:
            print(f"      [MHA] Re-merged & projected output shape:   {list(out.shape)}")

        return out

# ─────────────────────────────────────────────────────────────
# 3. Position-Wise Feed-Forward Module
# ─────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x, verbose=False):
        expanded = F.relu(self.w1(x))
        if verbose:
            print(f"      [FFN] Expanded to d_ff=2048:               {list(expanded.shape)}")
        projected = self.w2(expanded)
        if verbose:
            print(f"      [FFN] Projected back to d_model=512:        {list(projected.shape)}")
        return projected

# ─────────────────────────────────────────────────────────────
# 4. Encoder Block with Add & Norm
# ─────────────────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, verbose=False):
        if verbose:
            print("\n  --- [Inside Encoder Block] ---")
        # Sublayer 1: Attention + Add & Norm
        attn_out = self.attention(x, verbose=verbose)
        x = self.norm1(x + attn_out)

        # Sublayer 2: FeedForward + Add & Norm
        ffn_out = self.feed_forward(x, verbose=verbose)
        x = self.norm2(x + ffn_out)

        return x

# ─────────────────────────────────────────────────────────────
# 5. Full Encoder Stack Execution & Step-by-Step Print
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    # Example sentence: "How are you today" -> 4 token IDs
    tokens = torch.tensor([[42, 105, 12, 809]])  # Batch=1, Length=4
    print("================ DATA FLOW TRACER ================")
    print(f"Step 0: Input Token IDs:                   {tokens.tolist()}")
    print(f"        Shape:                            {list(tokens.shape)}  [Batch, Seq_Len]")

    # 1. Embedding lookup table
    vocab_size = 1000
    d_model = 512
    embedding = nn.Embedding(vocab_size, d_model)
    x_emb = embedding(tokens)
    print(f"\nStep 1: Learned Word Embeddings (E):       {list(x_emb.shape)} [Batch, Seq_Len, d_model]")

    # 2. Add Positional Encoding
    pos_encoder = PositionalEncoding(d_model=d_model)
    x = pos_encoder(x_emb)
    print(f"Step 2: After Positional Encoding (E + PE): {list(x.shape)} [Batch, Seq_Len, d_model]")

    # 3. Pass through Encoder Block
    block = EncoderBlock(d_model=d_model, n_heads=8, d_ff=2048)
    final_output = block(x, verbose=True)

    print("\n================ FINAL SUMMARY ================")
    print(f"Final Encoder Block Output Shape:          {list(final_output.shape)} [Batch, Seq_Len, d_model]")