"""Extract mel features for every indexed song -> cache dir + labels.pkl.

Env vars:
  STEPMANAI_INDEX   input index.json (from scan_library.py)
  STEPMANAI_CACHE   output dir for per-song .npy features
  STEPMANAI_LABELS  output labels.pkl
  STEPMANAI_U8      "1" = store uint8-quantized features (scale 10/255)
  STEPMANAI_WORKERS process count (default: cpu count)
"""
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX = os.environ.get("STEPMANAI_INDEX", os.path.join(ROOT, "data", "index.json"))
CACHE = os.environ.get("STEPMANAI_CACHE", os.path.join(ROOT, "data", "cache"))
LABELS = os.environ.get("STEPMANAI_LABELS", os.path.join(ROOT, "data", "labels.pkl"))
U8 = os.environ.get("STEPMANAI_U8", "0") == "1"
U8_SCALE = 10.0 / 255.0


def song_id(sim_path: str) -> str:
    return hashlib.md5(sim_path.encode("utf-8")).hexdigest()[:16]


def process(entry):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from smai.audio import features, load_audio
    from smai.simfile import parse_simfile

    sid = song_id(entry["sim"])
    npy = os.path.join(CACHE, sid + ".npy")
    try:
        if not os.path.exists(npy):
            x = load_audio(entry["music"])
            if len(x) < 16000 * 20:
                return sid, None, "too short"
            feat = features(x)
            if U8:
                feat = np.clip(np.round(feat.astype(np.float32) / U8_SCALE),
                               0, 255).astype(np.uint8)
            np.save(npy, feat)
            n_frames = feat.shape[0]
        else:
            n_frames = np.load(npy, mmap_mode="r").shape[0]
        song = parse_simfile(entry["sim"])
        charts = []
        for c in song.charts:
            steps = [(n.time, n.beat, n.row) for n in c.notes]
            if len([s for s in steps if any(ch in "124" for ch in s[2])]) < 20:
                continue
            charts.append({"diff": c.difficulty, "meter": c.meter, "steps": steps})
        if not charts:
            return sid, None, "no charts"
        return sid, {"sim": entry["sim"], "title": entry["title"],
                     "n_frames": n_frames, "charts": charts}, None
    except Exception as e:
        return sid, None, repr(e)


def main():
    with open(INDEX, encoding="utf-8") as f:
        index = json.load(f)
    os.makedirs(CACHE, exist_ok=True)
    workers = int(os.environ.get("STEPMANAI_WORKERS", str(os.cpu_count() or 2)))
    labels = {}
    errs = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, e) for e in index]
        for i, fut in enumerate(as_completed(futs)):
            sid, lab, err = fut.result()
            if lab:
                labels[sid] = lab
            else:
                errs += 1
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"{i+1}/{len(index)} ({el:.0f}s, {errs} errs)", flush=True)
    os.makedirs(os.path.dirname(LABELS), exist_ok=True)
    with open(LABELS, "wb") as f:
        pickle.dump(labels, f)
    print(f"DONE {len(labels)} songs cached, {errs} errors, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
