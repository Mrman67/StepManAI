"""Train the step selection model (which arrows, given rhythm + difficulty)."""
import json
import os
import pickle
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from smai.models import SelectionModel, beat_frac_onehot, hold_bucket
from smai.simfile import DIFF_ORDER

ROOT = os.environ.get("STEPMANAI_ROOT", os.path.dirname(os.path.abspath(__file__)))
DIFF_IDX = {d: i for i, d in enumerate(DIFF_ORDER)}
MAX_LEN = 512


def normalize_row(row: str) -> str:
    out = []
    for ch in row[:4]:
        if ch == "1":
            out.append("1")
        elif ch in "24":
            out.append("2")
        else:
            out.append("0")
    return "".join(out)


def chart_sequence(steps):
    """steps: [(time, beat, raw_row)] -> (tokens, times, beats, hold_buckets)."""
    toks, times, beats, holds = [], [], [], []
    # pre-index hold ends per panel
    for i, (t, b, row) in enumerate(steps):
        tok = normalize_row(row)
        if tok == "0000":
            continue
        hb = -100
        if "2" in tok:
            # find matching '3' for the longest hold in this row
            best = 0.0
            for p in range(4):
                if tok[p] == "2":
                    for t2, b2, row2 in steps[i + 1:]:
                        if len(row2) > p and row2[p] == "3":
                            best = max(best, b2 - b)
                            break
            if best > 0:
                hb = hold_bucket(best)
            else:
                tok = tok.replace("2", "1")     # unterminated hold -> tap
        toks.append(tok)
        times.append(t)
        beats.append(b)
        holds.append(hb)
    return toks, times, beats, holds


def reduce_token(tok: str, vocab: dict) -> str:
    if tok in vocab:
        return tok
    t2 = tok.replace("2", "1")
    if t2 in vocab:
        return t2
    # keep at most 2 panels
    on = [i for i, c in enumerate(t2) if c != "0"]
    if len(on) > 2:
        t3 = "".join("1" if i in on[:2] else "0" for i in range(4))
        if t3 in vocab:
            return t3
    for i, c in enumerate(t2):
        if c != "0":
            return "".join("1" if j == i else "0" for j in range(4))
    return "1000"


class SelDataset(Dataset):
    def __init__(self, seqs, vocab):
        self.seqs = seqs
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        toks, times, beats, holds, diff, meter = self.seqs[i]
        n = len(toks)
        ids = [self.vocab[reduce_token(t, self.vocab)] for t in toks]
        prev = [len(self.vocab)] + ids[:-1]          # BOS
        feats = np.zeros((n, 7), dtype=np.float32)
        for j in range(n):
            dtp = times[j] - times[j - 1] if j > 0 else 1.0
            dtn = times[j + 1] - times[j] if j < n - 1 else 1.0
            feats[j, 0] = np.log1p(max(dtp, 0.0) * 10)
            feats[j, 1] = np.log1p(max(dtn, 0.0) * 10)
            feats[j, 2:] = beat_frac_onehot(beats[j])
        return (torch.tensor(prev), torch.tensor(feats),
                torch.tensor(ids), torch.tensor(holds),
                diff, meter)


def collate(batch):
    prevs, feats, ids, holds, diffs, meters = zip(*batch)
    return (pad_sequence(prevs, batch_first=True, padding_value=0),
            pad_sequence(feats, batch_first=True),
            pad_sequence(ids, batch_first=True, padding_value=-100),
            pad_sequence(holds, batch_first=True, padding_value=-100),
            torch.tensor(diffs), torch.tensor(meters))


def main():
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    with open(os.environ.get("STEPMANAI_LABELS",
                             os.path.join(ROOT, "data", "labels.pkl")), "rb") as f:
        labels = pickle.load(f)
    sids = sorted(labels)
    random.Random(42).shuffle(sids)
    val_sids = set(sids[:max(30, len(sids) // 20)])

    def make_seqs(sel):
        seqs = []
        for sid in sel:
            for c in labels[sid]["charts"]:
                toks, times, beats, holds = chart_sequence(c["steps"])
                if len(toks) < 16:
                    continue
                d, m = DIFF_IDX[c["diff"]], min(c["meter"], 19)
                for s in range(0, len(toks), MAX_LEN):
                    part = slice(s, s + MAX_LEN)
                    if len(toks[part]) >= 16:
                        seqs.append((toks[part], times[part], beats[part],
                                     holds[part], d, m))
        return seqs

    train_seqs = make_seqs([s for s in sids if s not in val_sids])
    val_seqs = make_seqs(val_sids)

    counts = Counter(t for s in train_seqs for t in s[0])
    vocab_toks = sorted([t for t, n in counts.items() if n >= 30])
    vocab = {t: i for i, t in enumerate(vocab_toks)}
    print(f"train seqs {len(train_seqs)}, val {len(val_seqs)}, vocab {len(vocab)}",
          flush=True)
    with open(os.path.join(ROOT, "checkpoints", "sel_vocab.json"), "w") as f:
        json.dump(vocab_toks, f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SelectionModel(vocab=len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = int(os.environ.get("EPOCHS", "30"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_dl = DataLoader(SelDataset(train_seqs, vocab), batch_size=48,
                          shuffle=True, num_workers=2, collate_fn=collate,
                          persistent_workers=True, drop_last=True)
    val_dl = DataLoader(SelDataset(val_seqs, vocab), batch_size=48,
                        shuffle=False, num_workers=1, collate_fn=collate)

    best = 1e9
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = cnt = 0.0
        for prev, feats, ids, holds, diff, meter in train_dl:
            prev, feats, ids = prev.to(device), feats.to(device), ids.to(device)
            holds, diff, meter = holds.to(device), diff.to(device), meter.to(device)
            logits, hold_logits, _ = model(prev, feats, diff, meter)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), ids.reshape(-1),
                ignore_index=-100)
            hl = nn.functional.cross_entropy(
                hold_logits.reshape(-1, hold_logits.shape[-1]), holds.reshape(-1),
                ignore_index=-100)
            loss = loss + 0.3 * hl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            cnt += 1
        sched.step()

        model.eval()
        vtot = vcnt = acc = accn = 0.0
        with torch.no_grad():
            for prev, feats, ids, holds, diff, meter in val_dl:
                prev, feats, ids = prev.to(device), feats.to(device), ids.to(device)
                holds, diff, meter = holds.to(device), diff.to(device), meter.to(device)
                logits, hold_logits, _ = model(prev, feats, diff, meter)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), ids.reshape(-1),
                    ignore_index=-100)
                vtot += loss.item()
                vcnt += 1
                mask = ids >= 0
                acc += (logits.argmax(-1)[mask] == ids[mask]).sum().item()
                accn += mask.sum().item()
        vloss = vtot / vcnt
        print(f"ep {ep+1}/{epochs} train {tot/cnt:.4f} val {vloss:.4f} "
              f"acc {acc/accn:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vloss < best:
            best = vloss
            torch.save({"model": model.state_dict(), "vocab": vocab_toks,
                        "val": vloss, "epoch": ep},
                       os.path.join(ROOT, "checkpoints", "selection.pt"))
            print("  saved", flush=True)
    print("BEST", best)


if __name__ == "__main__":
    main()
