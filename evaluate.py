import torch
import sacrebleu

from config import CHECKPOINT_PATH, DEVICE
from vocabulary import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SOS_IDX, EOS_IDX
from utils import make_src_mask, make_trg_mask


@torch.no_grad()
def beam_search_decode(model, src, src_mask, beam_size: int = 5, max_len: int = 100, alpha: float = 0.6):
    model.eval()

    enc_out = model.encode(src, src_mask)
    beams = [{
        'tokens'  : torch.tensor([[SOS_IDX]], dtype=torch.long, device=DEVICE),
        'log_prob': 0.0,
        'done'    : False,
    }]
    completed = []

    for _ in range(max_len):
        if not beams:
            break
        candidates = []
        for beam in beams:
            if beam['done']:
                candidates.append(beam)
                continue
            trg = beam['tokens']
            trg_mask  = make_trg_mask(trg)
            logits    = model.project(model.decode(enc_out, src_mask, trg, trg_mask))
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            topk_log_probs, topk_tokens = log_probs[0].topk(beam_size)
            for lp, tok in zip(topk_log_probs.tolist(), topk_tokens.tolist()):
                new_tokens   = torch.cat([trg, torch.tensor([[tok]], dtype=torch.long, device=DEVICE)], dim=1)
                new_log_prob = beam['log_prob'] + lp
                done         = (tok == EOS_IDX)
                candidates.append({
                    'tokens'  : new_tokens,
                    'log_prob': new_log_prob,
                    'done'    : done,
                })

        def score(b):
            length = b['tokens'].size(1)
            return b['log_prob'] / (length ** alpha)

        candidates.sort(key=score, reverse=True)
        beams = []
        for c in candidates:
            if c['done']:
                completed.append(c)
            else:
                beams.append(c)
            if len(beams) >= beam_size:
                break

    all_hyps = completed if completed else beams
    best     = max(all_hyps, key=score)
    tokens   = best['tokens'][0].tolist()
    if tokens[0] == SOS_IDX:
        tokens = tokens[1:]
    if EOS_IDX in tokens:
        tokens = tokens[:tokens.index(EOS_IDX)]
    return tokens


def decode_tokens(indices, vocab) -> str:
    words = []
    for i in indices:
        i = i.item() if hasattr(i, 'item') else i
        if i == EOS_IDX:
            break
        w = vocab.idx2word.get(i, UNK_TOKEN)
        if w not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
            words.append(w)
    return ' '.join(words)


@torch.no_grad()
def compute_bleu_beam(model, dataset, trg_vocab, beam_size: int = 5, alpha: float = 0.6, max_samples=None):
    model.eval()
    hypotheses, references = [], []
    n = max_samples or len(dataset)

    for idx in range(min(n, len(dataset))):
        src_tensor, trg_tensor = dataset[idx]
        src_tensor = src_tensor.unsqueeze(0).to(DEVICE)
        src_mask   = make_src_mask(src_tensor)
        pred_ids   = beam_search_decode(model, src_tensor, src_mask, beam_size=beam_size, alpha=alpha)
        pred_str   = ' '.join(trg_vocab.idx2word.get(i, UNK_TOKEN) for i in pred_ids)
        ref_str    = decode_tokens(trg_tensor.tolist(), trg_vocab)
        hypotheses.append(pred_str)
        references.append(ref_str)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score, hypotheses, references


def run_evaluation(model, val_ds, test_ds, trg_vocab):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    val_bleu, val_hyp, val_ref = compute_bleu_beam(model, val_ds, trg_vocab, beam_size=5)
    print(f'  Val BLEU: {val_bleu:.2f}')

    test_bleu, test_hyp, test_ref = compute_bleu_beam(model, test_ds, trg_vocab, beam_size=5)
    print(f'  Test BLEU: {test_bleu:.2f}')

    print('=' * 70)
    print('TRANSLATIONS - TEST SET')
    print('=' * 70)
    for i in range(10):
        print(f'\n[{i+1}]')
        print(f'  REF : {test_ref[i]}')
        print(f'  PRED: {test_hyp[i]}')

    return val_bleu, test_bleu
