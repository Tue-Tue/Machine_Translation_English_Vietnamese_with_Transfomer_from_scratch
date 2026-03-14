import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from config import (SEED, MAX_TRAIN, MAX_SRC_LEN, MAX_TRG_LEN,
                    D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT,
                    N_EPOCHS, LR, CLIP, CHECKPOINT_PATH, DEVICE)
from vocabulary import Vocabulary
from dataset import build_dataloaders, find_file, read_lines
from model import build_transformer
from utils import make_src_mask, make_trg_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    for src, trg in loader:
        src, trg   = src.to(DEVICE), trg.to(DEVICE)
        src_mask   = make_src_mask(src)
        trg_input  = trg[:, :-1]
        trg_target = trg[:, 1:]
        trg_mask   = make_trg_mask(trg_input)
        logits = model(src, trg_input, src_mask, trg_mask)
        loss   = criterion(logits.reshape(-1, logits.size(-1)), trg_target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for src, trg in loader:
        src, trg   = src.to(DEVICE), trg.to(DEVICE)
        src_mask   = make_src_mask(src)
        trg_input  = trg[:, :-1]
        trg_target = trg[:, 1:]
        trg_mask   = make_trg_mask(trg_input)
        logits     = model(src, trg_input, src_mask, trg_mask)
        total_loss += criterion(logits.reshape(-1, logits.size(-1)), trg_target.reshape(-1)).item()
    return total_loss / len(loader)


def plot_losses(train_losses, val_losses):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-o', label='Train')
    axes[0].plot(epochs, val_losses,   'r-o', label='Val')
    axes[0].set(title='Loss', xlabel='Epoch', ylabel='CE Loss')
    axes[0].legend()
    axes[1].plot(epochs, [math.exp(l) for l in val_losses], 'g-o')
    axes[1].set(title='Val Perplexity', xlabel='Epoch', ylabel='PPL')
    plt.tight_layout()
    plt.show()


def main():
    set_seed(SEED)

    train_en_path = find_file('train.en.txt')
    train_vn_path = find_file('train.vi.txt')
    train_en = read_lines(train_en_path, MAX_TRAIN)
    train_vn = read_lines(train_vn_path, MAX_TRAIN)

    src_vocab = Vocabulary(min_freq=2)
    src_vocab.build(train_en)
    trg_vocab = Vocabulary(min_freq=2)
    trg_vocab.build(train_vn)

    (_, val_ds, _), (train_loader, val_loader, _), _ = build_dataloaders(src_vocab, trg_vocab, MAX_TRAIN)

    model = build_transformer(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        src_seq_len=MAX_SRC_LEN,
        trg_seq_len=MAX_TRG_LEN,
        d_model=D_MODEL, Nx=N_LAYERS, h=N_HEADS, dropout=DROPOUT, d_ff=D_FF
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=N_EPOCHS,
        pct_start=0.05
    )

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler)
        vl_loss = evaluate(model, val_loader, criterion)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        elapsed = time.time() - t0
        print(f'Epoch {epoch:2d}/{N_EPOCHS} | '
              f'train={tr_loss:.4f} | val={vl_loss:.4f} | '
              f'ppl={math.exp(vl_loss):.2f} | lr={scheduler.get_last_lr()[0]:.2e} | '
              f'{elapsed:.0f}s')
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print('saved best model')

    plot_losses(train_losses, val_losses)


if __name__ == '__main__':
    main()
