"""Scan a Songs library, validate charts, write the song index.

Env vars: STEPMANAI_SONGS (songs root), STEPMANAI_INDEX (output json).
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from smai.simfile import find_simfiles, parse_simfile

SONGS = os.environ.get("STEPMANAI_SONGS", r"C:\Games\OutFox 0.5.0 Alpha Win64\Songs")
INDEX = os.environ.get("STEPMANAI_INDEX",
                       os.path.join(os.path.dirname(__file__), "data", "index.json"))

t0 = time.time()
files = find_simfiles(SONGS)
print(f"{len(files)} simfiles found under {SONGS}")

diff_count = Counter()
densities = defaultdict(list)   # meter -> notes per second
index = []
errors = 0
no_audio = 0
for f in files:
    try:
        song = parse_simfile(f)
    except Exception as e:
        print("ERR", f, repr(e))
        errors += 1
        continue
    if song is None:
        errors += 1
        continue
    if not song.music or not os.path.exists(song.music):
        no_audio += 1
        continue
    entry = {"sim": f, "music": song.music, "title": song.title,
             "offset": song.offset, "charts": []}
    for c in song.charts:
        steps = c.step_rows()
        if len(steps) < 20:
            continue
        dur = steps[-1].time - steps[0].time
        if dur <= 10 or steps[0].time < -1:
            continue
        nps = len(steps) / dur
        diff_count[c.difficulty] += 1
        densities[c.meter].append(nps)
        entry["charts"].append({"diff": c.difficulty, "meter": c.meter,
                                "nsteps": len(steps), "nps": round(nps, 3)})
    if entry["charts"]:
        index.append(entry)

print(f"parsed in {time.time()-t0:.1f}s, {errors} errors, {no_audio} missing audio")
print(f"{len(index)} usable songs, {sum(len(e['charts']) for e in index)} charts")
for d, n in diff_count.most_common():
    print(f"  {d:10s} {n:5d} charts")
print("meter -> mean notes/sec:")
for m in sorted(densities):
    v = densities[m]
    print(f"  {m:3d}: {sum(v)/len(v):.2f} nps  ({len(v)} charts)")

os.makedirs(os.path.dirname(INDEX), exist_ok=True)
with open(INDEX, "w", encoding="utf-8") as f:
    json.dump(index, f)
print("wrote", INDEX)
