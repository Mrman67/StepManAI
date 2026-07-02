"""Train the step placement model (when do steps happen, given audio + difficulty)."""
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from smai.models import PlacementModel
from smai.simfile import DIFF_ORDER

ROOT = os.environ.get("STEPMANAI_ROOT", os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("STEPMANAI_CACHE", os.path.join(ROOT, "data", "cache"))
FPS = 100
U8_SCALE = 10.0 / 255.0     # dequant scale for uint8-packed features
CROP = 768
DIFF_IDX = {d: i for i, d in enumerate(DIFF_ORDER)}


class PlacementDataset(Dataset):
    def __init__(self, items):
        self.items = items      # (sid, n_frames, diff_idx, meter, step_frames)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, n_frames, diff, meter, frames = self.items[i]
        feat = np.load(os.path.join(CACHE, sid + ".npy"), mmap_mode="r")
        n = min(n_frames, feat.shape[0])
        if n <= CROP:
            start = 0
        else:
            # bias crops towards the charted region
            lo = max(0, int(frames[0]) - 100)
            hi = min(n - CROP, max(lo + 1, int(frames[-1]) - CROP + 100))
            start = random.randint(lo, hi) if hi > lo else 0
        x = np.zeros((CROP, 3, 80), dtype=np.float32)
        seg = np.asarray(feat[start:start + CROP])
        if seg.dtype == np.uint8:
            seg = seg.astype(np.float32) * U8_SCALE
        x[:seg.shape[0]] = seg.astype(np.float32)
        y = np.zeros(CROP, dtype=np.float32)
        w = np.ones(CROP, dtype=np.float32)
        w[seg.shape[0]:] = 0.0
        for f in frames:
            f = int(f) - start
            if 0 <= f < CROP:
                y[f] = 1.0
                for d in (-1, 1):
                    if 0 <= f + d < CROP:
                        y[f + d] = max(y[f + d], 0.5)
        return x, y, w, diff, meter


def build_items(labels):
    items = []
    for sid, lab in labels.items():
        for c in lab["charts"]:
            steps = [s for s in c["steps"] if any(ch in "124" for ch in s[2])]
            if len(steps) < 20:
                continue
            frames = sorted({round(t * FPS) for t, _, _ in
                             [(s[0], s[1], s[2]) for s in steps] if t >= 0})
            frames = [f for f in frames if f < lab["n_frames"]]
            if len(frames) < 20:
                continue
            items.append((sid, lab["n_frames"], DIFF_IDX[c["diff"]],
                          min(c["meter"], 19), np.asarray(frames, dtype=np.int64)))
    return items


def f1_at(logits, targets, weights, thresh=0.0, tol=2):
    """Greedy onset F1 with +-tol frame tolerance, targets = exact frames (y==1)."""
    probs = logits
    tp = fp = fn = 0
    for b in range(logits.shape[0]):
        pred = []
        p = probs[b]
        for i in range(1, p.shape[0] - 1):
            if weights[b, i] > 0 and p[i] > thresh and p[i] >= p[i - 1] and p[i] >= p[i + 1]:
                pred.append(i)
        true = [i for i in range(p.shape[0]) if targets[b, i] == 1.0]
        used = set()
        for t in true:
            hit = None
            for pr in pred:
                if pr not in used and abs(pr - t) <= tol:
                    hit = pr
                    break
            if hit is not None:
                used.add(hit)
                tp += 1
            else:
                fn += 1
        fp += len(pred) - len(used)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


def main():
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    with open(os.environ.get("STEPMANAI_LABELS",
                             os.path.join(ROOT, "data", "labels.pkl")), "rb") as f:
        labels = pickle.load(f)
    sids = sorted(labels)
    random.Random(42).shuffle(sids)
    n_val = max(30, len(sids) // 20)
    val_sids = set(sids[:n_val])
    train_items = build_items({s: labels[s] for s in sids if s not in val_sids})
    val_items = build_items({s: labels[s] for s in val_sids})
    print(f"train charts {len(train_items)}, val charts {len(val_items)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    model = PlacementModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = int(os.environ.get("EPOCHS", "40"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    pos_weight = torch.tensor(6.0, device=device)

    batch = int(os.environ.get("BATCH", "24"))
    train_dl = DataLoader(PlacementDataset(train_items), batch_size=batch,
                          shuffle=True, num_workers=3, pin_memory=(device == "cuda"),
                          persistent_workers=True, drop_last=True)
    val_dl = DataLoader(PlacementDataset(val_items), batch_size=batch,
                        shuffle=False, num_workers=2)

    best = 0.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = cnt = 0.0
        for x, y, w, diff, meter in train_dl:
            x, y, w = x.to(device), y.to(device), w.to(device)
            diff, meter = diff.to(device), meter.to(device)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                logits = model(x, diff, meter)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y, weight=w, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            cnt += 1
        sched.step()

        model.eval()
        f1s = []
        vloss = vcnt = 0.0
        with torch.no_grad():
            for x, y, w, diff, meter in val_dl:
                x, y, w = x.to(device), y.to(device), w.to(device)
                diff, meter = diff.to(device), meter.to(device)
                with torch.amp.autocast(device, enabled=(device == "cuda")):
                    logits = model(x, diff, meter)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, y, weight=w, pos_weight=pos_weight)
                vloss += loss.item()
                vcnt += 1
                f1, p, r = f1_at(logits.float().cpu().numpy(), y.cpu().numpy(),
                                 w.cpu().numpy())
                f1s.append(f1)
        vf1 = sum(f1s) / len(f1s)
        print(f"ep {ep+1}/{epochs} train {tot/cnt:.4f} val {vloss/vcnt:.4f} "
              f"F1 {vf1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best:
            best = vf1
            torch.save({"model": model.state_dict(), "f1": vf1, "epoch": ep},
                       os.path.join(ROOT, "checkpoints", "placement.pt"))
            print(f"  saved (F1 {vf1:.3f})", flush=True)
    print("BEST F1", best)


if __name__ == "__main__":
    main()
