# English → Vietnamese Neural Machine Translation

Transformer model built **from scratch** with PyTorch — no Hugging Face, no pre-built transformer layers.  
Trained on the [IWSLT'15 English-Vietnamese](https://www.kaggle.com/datasets/tuannguyenvananh/iwslt15-englishvietnamese) dataset.

## Results

| Split | BLEU |
|-------|------|
| Val (tst2012) | *fill in* |
| Test (tst2013) | *fill in* |

## Architecture

- Encoder-Decoder Transformer (4 layers, 8 heads, d_model=256, d_ff=1024)
- Custom Multi-Head Attention, Positional Encoding, Layer Normalization
- Beam Search decoding (beam size=5, length penalty α=0.6)
- Label Smoothing (0.1) + OneCycleLR scheduler

## Project Structure

```
├── config.py        # Hyperparameters & paths
├── vocabulary.py    # Tokenizer & Vocabulary class
├── dataset.py       # TranslationDataset & DataLoaders
├── model.py         # Full Transformer architecture
├── utils.py         # Mask helpers
├── train.py         # Training loop
├── evaluate.py      # Beam search & BLEU scoring
├── translate.py     # Interactive inference demo
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Update `DATA_DIR` in `config.py` to point to your local dataset path.

## Training

```bash
python train.py
```

## Evaluation

```python
from evaluate import run_evaluation
run_evaluation(model, val_ds, test_ds, trg_vocab)
```

## Inference

```bash
python translate.py
```
