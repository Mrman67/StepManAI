"""Quantize the feature cache to uint8 and tar it for Colab training.

Output: data\stepmanai_dataset.tar containing
  cache_u8/<sid>.npy   (T,3,80) uint8, value = round(logmel / (10/255))
  labels.pkl
"""
import io
import os
import pickle
import sys
import tarfile
import time

import numpy as np

ROOT = r"C:\StepManAI"
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "data", "stepmanai_dataset.tar")
U8_SCALE = 10.0 / 255.0

with open(os.path.join(ROOT, "data", "labels.pkl"), "rb") as f:
    labels = pickle.load(f)
print(f"{len(labels)} songs in labels")

t0 = time.time()
with tarfile.open(OUT, "w") as tar:
    lb = pickle.dumps(labels)
    info = tarfile.TarInfo("labels.pkl")
    info.size = len(lb)
    tar.addfile(info, io.BytesIO(lb))
    for i, sid in enumerate(sorted(labels)):
        p = os.path.join(CACHE, sid + ".npy")
        feat = np.load(p).astype(np.float32)
        q = np.clip(np.round(feat / U8_SCALE), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        np.save(buf, q)
        data = buf.getvalue()
        info = tarfile.TarInfo(f"cache_u8/{sid}.npy")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(labels)} ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE -> {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB, {time.time()-t0:.0f}s)")
