from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vocabulary import tokenize, Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX
from config import DATA_DIR, MAX_SRC_LEN, MAX_TRG_LEN, BATCH_SIZE, NUM_WORKERS


def find_file(name: str) -> Path:
    p = DATA_DIR / name
    if p.exists():
        return p
    for sub in DATA_DIR.iterdir():
        if sub.is_dir() and (sub / name).exists():
            return sub / name


def read_lines(path, max_lines=None) -> list:
    with open(path, encoding='utf-8') as f:
        lines = [l.strip() for l in f]
    return lines[:max_lines] if max_lines else lines


class TranslationDataset(Dataset):
    def __init__(self, src_sents, trg_sents, src_vocab: Vocabulary, trg_vocab: Vocabulary,
                 max_src=100, max_trg=100):
        pairs = [
            (s, t) for s, t in zip(src_sents, trg_sents)
            if len(tokenize(s)) <= max_src - 2
            and len(tokenize(t)) <= max_trg - 2
        ]
        self.src = [p[0] for p in pairs]
        self.trg = [p[1] for p in pairs]
        self.sv  = src_vocab
        self.tv  = trg_vocab
        print(f'{len(self.src):,} pairs')

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        s = torch.tensor([SOS_IDX] + self.sv.encode(self.src[idx]) + [EOS_IDX], dtype=torch.long)
        t = torch.tensor([SOS_IDX] + self.tv.encode(self.trg[idx]) + [EOS_IDX], dtype=torch.long)
        return s, t


def collate_fn(batch):
    src, trg = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    trg = nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=PAD_IDX)
    return src, trg


def build_dataloaders(src_vocab: Vocabulary, trg_vocab: Vocabulary, max_train=None):
    TRAIN_EN = find_file('train.en.txt')
    TRAIN_VN = find_file('train.vi.txt')
    VAL_EN   = find_file('tst2012.en.txt')
    VAL_VN   = find_file('tst2012.vi.txt')
    TEST_EN  = find_file('tst2013.en.txt')
    TEST_VN  = find_file('tst2013.vi.txt')

    train_en = read_lines(TRAIN_EN, max_train)
    train_vn = read_lines(TRAIN_VN, max_train)
    val_en   = read_lines(VAL_EN)
    val_vn   = read_lines(VAL_VN)
    test_en  = read_lines(TEST_EN)
    test_vn  = read_lines(TEST_VN)

    print('Train:'); train_ds = TranslationDataset(train_en, train_vn, src_vocab, trg_vocab, MAX_SRC_LEN, MAX_TRG_LEN)
    print('Val:');   val_ds   = TranslationDataset(val_en,   val_vn,   src_vocab, trg_vocab, MAX_SRC_LEN, MAX_TRG_LEN)
    print('Test:');  test_ds  = TranslationDataset(test_en,  test_vn,  src_vocab, trg_vocab, MAX_SRC_LEN, MAX_TRG_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    return (train_ds, val_ds, test_ds), (train_loader, val_loader, test_loader), (train_en, train_vn)
