"""Scan the OutFox Songs library, validate the parser, dump stats + index."""
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from smai.simfile import find_simfiles, parse_simfile

SONGS = r"C:\Games\OutFox 0.5.0 Alpha Win64\Songs"

t0 = time.time()
files = find_simfiles(SONGS)
print(f"{len(files)} simfiles found")

diff_count = Counter()
meter_by_diff = defaultdict(list)
densities = defaultdict(list)   # meter -> notes per second
index = []
errors = 0
no_audio = 0
for i, f in enumerate(files):
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
        meter_by_diff[c.difficulty].append(c.meter)
        densities[c.meter].append(nps)
        entry["charts"].append({"diff": c.difficulty, "meter": c.meter,
                                "nsteps": len(steps), "nps": round(nps, 3)})
    if entry["charts"]:
        index.append(entry)

print(f"parsed in {time.time()-t0:.1f}s, {errors} errors, {no_audio} missing audio")
print(f"{len(index)} usable songs, {sum(len(e['charts']) for e in index)} charts")
for d, n in diff_count.most_common():
    ms = meter_by_diff[d]
    print(f"  {d:10s} {n:5d} charts, meter {min(ms)}-{max(ms)}")
print("meter -> mean notes/sec:")
for m in sorted(densities):
    v = densities[m]
    print(f"  {m:3d}: {sum(v)/len(v):.2f} nps  ({len(v)} charts)")

with open(r"C:\StepManAI\data\index.json", "w", encoding="utf-8") as f:
    json.dump(index, f)
print("wrote data/index.json")
