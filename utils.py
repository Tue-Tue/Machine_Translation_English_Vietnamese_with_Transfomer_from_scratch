import torch
from vocabulary import PAD_IDX


def make_src_mask(src: torch.Tensor) -> torch.Tensor:
    return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)


def make_trg_mask(trg: torch.Tensor) -> torch.Tensor:
    B, T = trg.size()
    pad_mask    = (trg != PAD_IDX).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(torch.ones(T, T, device=trg.device)).bool()
    return pad_mask & causal_mask
