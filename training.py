#tgt_input (Passed to Decoder): ["<SOS>", "हम", "दोस्त", "हैं"]
# [:-1] we never need EOS for input
#tgt_label (Expected Targets for Loss): ["हम", "दोस्त", "हैं", "<EOS>"]
#[1:] comparison from the first one

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TranslationDataset, collate_fn, PAD_IDX, SOS_IDX, EOS_IDX
from model import Transformer  # Uses the verified Vaswani Transformer from model.py

# ─────────────────────────────────────────────────────────────
# 1. Vaswani Learning Rate Scheduler (Section 5.3)
# ─────────────────────────────────────────────────────────────
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ─────────────────────────────────────────────────────────────
# 2. Single Epoch Training Function
# ─────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0

    for step, (src, tgt_in, tgt_label, src_mask, tgt_mask) in enumerate(dataloader):
        # 1. Push batch tensors to hardware accelerator
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_label = tgt_label.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        # 2. Forward pass: [B, T_tgt, tgt_vocab_size]
        logits = model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)

        # 3. Reshape for Cross-Entropy Loss:
        # [B, T, V] -> [B * T, V]
        # [B, T]    -> [B * T]
        vocab_size = logits.size(-1)
        logits_flat = logits.contiguous().view(-1, vocab_size)
        targets_flat = tgt_label.contiguous().view(-1)

        # 4. Compute loss (ignores PAD_IDX automatically)
        loss = criterion(logits_flat, targets_flat)

        # 5. Backward pass & parameter update
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step custom scheduler
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


