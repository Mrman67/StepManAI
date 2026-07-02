"""Extract mel features for every indexed song -> data/cache/*.npy + labels.pkl."""
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

ROOT = r"C:\StepManAI"
CACHE = os.path.join(ROOT, "data", "cache")


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
    with open(os.path.join(ROOT, "data", "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    os.makedirs(CACHE, exist_ok=True)
    labels = {}
    errs = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
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
    with open(os.path.join(ROOT, "data", "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)
    print(f"DONE {len(labels)} songs cached, {errs} errors, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
