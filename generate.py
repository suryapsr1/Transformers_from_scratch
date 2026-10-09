import torch
import torch.nn.functional as F
from model import Transformer
from dataset import create_padding_mask, create_target_mask, PAD_IDX, SOS_IDX, EOS_IDX

@torch.no_grad()
def greedy_decode(model, src, max_len=20, device="cpu"):
    """
    Autoregressive greedy decoding for a single source sentence.
    
    Args:
        model: Trained Transformer instance
        src: 2D Tensor of shape [1, T_src] containing source token IDs
        max_len: Safety cap to prevent infinite generation loops
        device: 'cpu' or 'cuda'
        
    Returns:
        List of generated target token IDs
    """
    model.eval()
    src = src.to(device)

    # 1. Source Padding Mask: [1, 1, 1, T_src]
    src_mask = create_padding_mask(src, pad_idx=PAD_IDX).to(device)

    # 2. Encode source sequence ONCE: [1, T_src, d_model]
    enc_out = model.encoder(src, src_mask=src_mask)

    # 3. Initialize target sequence with start token: [1, 1]
    tgt_tokens = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)

    for step in range(max_len):
        # Build causal + padding mask for target sequence: [1, 1, T_curr, T_curr]
        tgt_mask = create_target_mask(tgt_tokens, pad_idx=PAD_IDX).to(device)

        # Decoder forward pass using cached enc_out: [1, T_curr, d_model]
        dec_out = model.decoder(tgt_tokens, enc_out, tgt_mask=tgt_mask)

        # Project through vocabulary head: [1, T_curr, vocab_size]
        logits = model.fc_out(dec_out)

        # Focus exclusively on the very last predicted token position: [1, vocab_size]
        last_token_logits = logits[:, -1, :]

        # Greedy selection: pick the token with highest probability
        next_token_id = torch.argmax(last_token_logits, dim=-1).item()

        # Append predicted token to sequence: [1, T_curr + 1]
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        tgt_tokens = torch.cat([tgt_tokens, next_token_tensor], dim=1)

        # Stop early if End-of-Sequence token is generated
        if next_token_id == EOS_IDX:
            break

    # Return as clean list of integers
    return tgt_tokens.squeeze(0).tolist()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lightweight configuration matching train.py sanity check
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 128
    n_heads = 4
    d_ff = 512
    n_layers = 2

    # Instantiate model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers
    ).to(device)

    # Simulate an unseen test source sentence: [1, 4]
    test_src = torch.tensor([[15, 24, 8, 32]])

    # Run autoregressive greedy decoding
    generated_ids = greedy_decode(model, test_src, max_len=10, device=device)

    print("--- Autoregressive Decoding Verification ---")
    print("Source Token IDs:   ", test_src.squeeze(0).tolist())
    print("Generated Token IDs:", generated_ids)
    print("Generation complete!")




@torch.no_grad()
def beam_search_decode(model, src, beam_width=3, max_len=20, device="cpu"):
    """
    Beam search decoding keeping top-k hypothesis paths.
    
    Args:
        model: Trained Transformer instance
        src: 2D Tensor [1, T_src]
        beam_width: Number of active candidate hypotheses to maintain
        max_len: Max sequence length
        device: 'cpu' or 'cuda'
        
    Returns:
        List of generated target token IDs for the highest scoring path
    """
    model.eval()
    src = src.to(device)

    # 1. Encode source sequence ONCE
    src_mask = create_padding_mask(src, pad_idx=PAD_IDX).to(device)
    enc_out = model.encoder(src, src_mask=src_mask)

    # 2. Hypothesis pool: stores tuples of (token_tensor, cumulative_log_prob, is_done)
    # Start with initial prompt <SOS>
    initial_tokens = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
    hypotheses = [(initial_tokens, 0.0, False)]

    for step in range(max_len):
        candidates = []
        all_done = True

        for tokens, score, done in hypotheses:
            if done:
                candidates.append((tokens, score, True))
                continue

            all_done = False
            tgt_mask = create_target_mask(tokens, pad_idx=PAD_IDX).to(device)
            dec_out = model.decoder(tokens, enc_out, tgt_mask=tgt_mask)
            logits = model.fc_out(dec_out)

            # Convert step logits into log probabilities: [1, vocab_size]
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            # Get top-k candidates for this hypothesis path
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                next_token_id = topk_indices[0, i].item()
                next_score = score + topk_log_probs[0, i].item()

                next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                new_seq = torch.cat([tokens, next_tensor], dim=1)

                is_finished = (next_token_id == EOS_IDX)
                candidates.append((new_seq, next_score, is_finished))

        # Sort all candidates across all beams by cumulative log-prob descending
        ordered = sorted(candidates, key=lambda tup: tup[1], reverse=True)
        hypotheses = ordered[:beam_width]

        if all_done:
            break

    # Return the token IDs of the top-ranked hypothesis
    best_tokens, best_score, _ = hypotheses[0]
    return best_tokens.squeeze(0).tolist()