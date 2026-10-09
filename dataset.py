'''When we send data we need to make sure it is uniform, so we actually need one more mask here which will hide any <pad> token
it our model don't waste time on that null value'''


import torch

import torch

# Special Token IDs
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

seq = torch.tensor([
    [15, 24, 8, 0],   # Sentence 1: 3 words + 1 pad
    [4,   9, 0, 0]    # Sentence 2: 2 words + 2 pads
])

def create_padding_mask(seq, pad_idx=0):
    """
    seq shape: [Batch, Time] (e.g., [2, 4])
    Checks where tokens are NOT pad_idx (0).
    Output shape: [Batch, 1, 1, Time]
    """
    # True for valid words, False for <PAD>
    # vector operation of pytorch for simultanous comparison of everything return a tensor of same shape [2,4]
    #just data will be boolean
    mask = (seq != pad_idx)


    
    # Add two dimensions (Head, Query) so PyTorch can broadcast it over 4D attention scores [B, Heads, T_q, T_k]
    return mask.unsqueeze(1).unsqueeze(2)

#mask.unsqueeze(1) inserts a dimension at index 1:$$\text{Shape becomes } [B, \ 1, \ T]
# $$.unsqueeze(2) inserts a dimension at index 2:$$\text{Shape becomes } [B, \ 1, \ 1, \ T]$$


def create_target_mask(tgt_seq, pad_idx=0):
    """
    tgt_seq shape: [Batch, Time]
    Combines:
      1. Don't look at <PAD> tokens.
      2. Don't look at future tokens (lower-triangular).
    Output shape: [Batch, 1, Time, Time]
    """
    B, T = tgt_seq.size()
    
    # 1. Padding mask: [B, 1, 1, T]
    pad_mask = create_padding_mask(tgt_seq, pad_idx)
    
    # 2. Causal lower-triangular mask: [1, 1, T, T]
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt_seq.device)).unsqueeze(0).unsqueeze(0)
    
    # 3. Combine both with boolean AND (&)
    # A position is valid ONLY if it is NOT a PAD token AND it is in the past/present
    combined_mask = pad_mask & causal_mask
    return combined_mask


if __name__ == "__main__":
    print("Input Tensor (Token IDs):\n", seq)
    print("Input Shape:", seq.shape, "\n")

    # Generate source padding mask
    src_mask = create_padding_mask(seq, pad_idx=0)
    print("Source Padding Mask Shape:", src_mask.shape) # [2, 1, 1, 4]

    # Generate combined target mask
    tgt_mask = create_target_mask(seq, pad_idx=0)
    print("\nTarget Mask Shape:", tgt_mask.shape)       # [2, 1, 4, 4]
    
    # Inspect the 2D causal grid for Sentence 1
    print("\nTarget Mask Grid for Sentence 1 (0 = Blocked, 1 = Attended):")
    print(tgt_mask[0, 0].int())





# Shape: [2, 4]



if __name__ == "__main__":
    print("Input Tensor (Token IDs):\n", seq)
    print("Input Shape:", seq.shape, "\n")

    # Generate source padding mask
    src_mask = create_padding_mask(seq, pad_idx=0)
    print("Source Padding Mask Shape:", src_mask.shape) # [2, 1, 1, 4]

    # Generate combined target mask
    tgt_mask = create_target_mask(seq, pad_idx=0)
    print("\nTarget Mask Shape:", tgt_mask.shape)       # [2, 1, 4, 4]
    
    # Inspect the 2D causal grid for Sentence 1
    print("\nTarget Mask Grid for Sentence 1 (0 = Blocked, 1 = Attended):")
    print(tgt_mask[0, 0].int())