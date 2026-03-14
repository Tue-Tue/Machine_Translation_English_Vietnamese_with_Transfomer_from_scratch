import torch
from pathlib import Path

SEED = 42

DATA_DIR = Path("/kaggle/input/datasets/tuannguyenvananh/iwslt15-englishvietnamese/IWSLT'15 en-vi")

MAX_TRAIN   = None
MAX_SRC_LEN = 100
MAX_TRG_LEN = 100
BATCH_SIZE  = 128
NUM_WORKERS = 2

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
D_FF    = 1024
DROPOUT = 0.1

N_EPOCHS = 30
LR       = 1e-4
CLIP     = 1.0

CHECKPOINT_PATH = '/kaggle/working/best_model.pt'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
