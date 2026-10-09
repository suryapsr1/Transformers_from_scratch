import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Positional Encoding (Vaswani et al., Section 3.5)
# Injects sequential word-order information via fixed sinusoidal frequencies.
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Buffer moves automatically with the model across CPU/GPU without being a parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [B, T, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ==============================================================================
# 2. Multi-Head Attention (Vaswani et al., Section 3.2)
# Handles both Self-Attention (Q=K=V) and Cross-Attention (Q from Dec, K/V from Enc).
# ==============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 512 // 8 = 64

        # Linear projections for Query, Key, Value, and Output
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        # 1. Linear projections & split into 8 heads of 64
        # [B, T, 512] -> [B, 8, T, 64]
        q_proj = self.w_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product similarity scores: [B, 8, T_q, T_k]
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Mask out invalid or future positions (-1e9 turns to 0 in softmax)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax probability distribution over keys
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Weighted combination of values: [B, 8, T_q, 64]
        out = torch.matmul(attn_weights, v_proj)

        # 6. Recombine heads: [B, 8, T_q, 64] -> [B, T_q, 512]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # 7. Final output projection W^O
        return self.dropout(self.w_o(out))


# ==============================================================================
# 3. Position-Wise Feed-Forward Network (Vaswani et al., Section 3.3)
# Two linear layers with a ReLU activation: 512 -> 2048 -> 512.
# ==============================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''Makes x size of [2, 10, 512 ->2048] applies relu, convert back to 512'''
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ==============================================================================
# 4. Encoder Block & Stack (Vaswani et al., Section 3.1)
# 2 Sublayers: Self-Attention -> Feed-Forward.
# ==============================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Sublayer 1: Self-Attention (Q=x, K=x, V=x) + Residual + LayerNorm
        attn_out = self.self_attn(q=x, k=x, v=x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sublayer 2: Feed-Forward + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048, n_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
    '''Triggered by encoder call first'''
    def forward(self, src_tokens, src_mask=None):
        # [B, T_src] -> [B, T_src, d_model]
        '''calls position encoder which calls the embedding layer'''
        x = self.pos_encoder(self.embedding(src_tokens))
        '''Calls the encoder layer as defined in the __init__'''
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x


# ==============================================================================
# 5. Decoder Block & Stack (Vaswani et al., Section 3.1)
# 3 Sublayers: Masked Self-Attention -> CROSS-ATTENTION -> Feed-Forward.
# ==============================================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # Station 1: Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Station 2: CROSS-ATTENTION (Decoder Q meets Encoder K & V)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Station 3: Position-Wise Feed-Forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        # 1. Masked Self-Attention (Q=x, K=x, V=x)
        attn_out = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        '''The self attention alone have tgt mask which hides the next tokens during the 
        training time and faster training as parallel training but none in the cross
        attention as decoder can look at all the tokens from the encoder'''
        # 2. CROSS-ATTENTION: Query from Decoder (x), Keys/Values from Encoder (enc_out)
        cross_out = self.cross_attn(q=x, k=enc_out, v=enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. Feed-Forward Network
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048, n_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt_tokens, enc_out, tgt_mask=None, src_mask=None):
        # [B, T_tgt] -> [B, T_tgt, d_model]
        x = self.pos_encoder(self.embedding(tgt_tokens))
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)
        return x


# ==============================================================================
# 6. Complete Seq2Seq Transformer (Vaswani et al., Full Architecture)
# Stacks Encoder, Decoder, and the final Linear Vocabulary Projection Head.
# ==============================================================================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, d_ff=2048, n_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads, d_ff, n_layers, dropout)
        
        # Output Linear Head: Projects 512 dimensions -> target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 1. Run through 6 Encoder Blocks 
        '''Called second, calls encoder block'''
        enc_out = self.encoder(src, src_mask=src_mask)

        # 2. Run through 6 Decoder Blocks (Cross-Attention references enc_out)
        '''Third call, calls decoder block'''
        dec_out = self.decoder(tgt, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)

        # 3. Project to vocabulary logits: [B, T_tgt, tgt_vocab_size]
        logits = self.fc_out(dec_out)
        return logits


# 7. Helper: Causal Lower-Triangular Mask

def make_causal_mask(size):
    """Lower-triangular mask [1, 1, size, size] preventing attention to future tokens."""
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
'''This creates a matrix of ones first then, tril zeros out the upper traingle and creates a lower trianglar matrix
after which unsqueeze add more dimension ie encloses in one more bracket and then one more to make it as per the similarity score'''

#### Check run 
'''First step to be executed'''

if __name__ == "__main__":
    print("Testing Complete Seq2Seq Transformer...")

    batch_size = 2
    T_src = 10         # 10 tokens in English sentence
    T_tgt = 8          # 8 tokens in target language sentence
    src_vocab = 5000   # English vocabulary size
    tgt_vocab = 8000   # Target language vocabulary size

    # Simulated input batches

    '''torch.randint(low, high, size), create a tensor of size as required by the encoder [2, 10] 
       size sequence of random integer'''
    src_tokens = torch.randint(0, src_vocab, (batch_size, T_src))
    tgt_tokens = torch.randint(0, tgt_vocab, (batch_size, T_tgt))

    # Causal mask for decoder target sequence will call teh causal mask function, we target 8 token of the target sequence
    tgt_mask = make_causal_mask(T_tgt)

    # Instantiate the full Transformer (6 encoder layers + 6 decoder layers)
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        dropout=0.1
    )

    '''Calls transformer and the forward module runs'''

    logits = model(src_tokens, tgt_tokens, tgt_mask=tgt_mask)

    print("\n--- Tensor Shapes ---")
    print("Source tokens input: ", list(src_tokens.shape)) # [2, 10]
    print("Target tokens input: ", list(tgt_tokens.shape)) # [2, 8]
    print("Target mask shape:   ", list(tgt_mask.shape))   # [1, 1, 8, 8]
    print("Output Logits:       ", list(logits.shape))     # [2, 8, 8000]
    print("\nSuccess: The full Vaswani Transformer pipeline executed cleanly!")