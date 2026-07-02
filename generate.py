"""StepManAI chart generator.

Run with no arguments for interactive (form) mode:

  > python generate.py
  Audio file or Spotify link: https://open.spotify.com/track/XXXX
  Title  [auto from Spotify]:
  Artist [auto from Spotify]:
  Level (1-19, or 'all'):    12

Or non-interactive:
  python generate.py song.mp3 -t "Song Name" -a "Artist" -l 12
  python generate.py https://open.spotify.com/track/XXXX -l all

Level is the DDR difficulty number 1-19. 'all' makes a full 5-chart song
(levels 3/6/9/12/15). Output: a song folder under Songs\\StepManAI.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from smai.audio import FPS, features, load_audio
from smai.bpm import detect_bpm
from smai.models import HOLD_BUCKETS, PlacementModel, SelectionModel, beat_frac_onehot
from smai.simfile import DIFF_ORDER, write_sm

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("STEPMANAI_CKPT", os.path.join(ROOT, "checkpoints"))
OUTFOX_PACK = r"C:\Games\OutFox 0.5.0 Alpha Win64\Songs\StepManAI"
DIFF_IDX = {d: i for i, d in enumerate(DIFF_ORDER)}

# allowed beat-grid subdivisions per SM slot
SLOT_GRIDS = {
    "Beginner":  [1.0, 0.5],
    "Easy":      [1.0, 0.5],
    "Medium":    [1.0, 0.5, 0.25],
    "Hard":      [1.0, 0.5, 0.25, 1/3, 1/6],
    "Challenge": [1.0, 0.5, 0.25, 1/3, 1/6, 0.125],
}


def level_to_slot(level: int) -> str:
    """DDR level number -> SM difficulty slot (matches library meter ranges)."""
    if level <= 3:
        return "Beginner"
    if level <= 6:
        return "Easy"
    if level <= 9:
        return "Medium"
    if level <= 13:
        return "Hard"
    return "Challenge"


# 'all' generates these five (slot, level) charts
ALL_LEVELS = [("Beginner", 3), ("Easy", 6), ("Medium", 9),
              ("Hard", 12), ("Challenge", 15)]
# meter -> target notes/sec, fitted from the library scan
METER_NPS = {1: 0.71, 2: 0.86, 3: 1.14, 4: 1.43, 5: 1.81, 6: 2.14, 7: 2.46,
             8: 2.81, 9: 3.23, 10: 3.47, 11: 3.61, 12: 3.88, 13: 4.31,
             14: 4.74, 15: 5.23, 16: 5.85, 17: 6.34, 18: 7.00, 19: 8.16}

MAX_PANELS = {"Beginner": 2, "Easy": 2, "Medium": 2, "Hard": 2, "Challenge": 2}
JUMP_MIN_GAP = {"Beginner": 1.0, "Easy": 0.5, "Medium": 0.25,
                "Hard": 0.20, "Challenge": 0.15}   # seconds between jumps


def run_placement(model, feat, diff_i, meter, device):
    """Full-song per-frame probabilities, windowed inference."""
    T = feat.shape[0]
    win, ov = 4096, 256
    probs = np.zeros(T, dtype=np.float32)
    weight = np.zeros(T, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, T, win - 2 * ov):
            seg = np.asarray(feat[s:s + win], dtype=np.float32)
            x = torch.from_numpy(seg).unsqueeze(0).to(device)
            d = torch.tensor([diff_i], device=device)
            m = torch.tensor([min(meter, 19)], device=device)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(x, d, m)
            p = torch.sigmoid(logits.float()).cpu().numpy()[0]
            w = np.ones(len(p))
            if s > 0:
                w[:ov] = np.linspace(0, 1, ov)
            if s + win < T:
                w[-ov:] = np.linspace(1, 0, ov)
            probs[s:s + len(p)] += p * w
            weight[s:s + len(p)] += w
            if s + win >= T:
                break
    return probs / np.maximum(weight, 1e-9)


def pick_steps(probs, bpm, t0, grids, target_count, min_gap_beats):
    """Snap placement probabilities to the beat grid, take best target_count."""
    spb = 60.0 / bpm
    dur = len(probs) / FPS
    # candidate grid positions over the whole song
    cands = set()
    n_beats = int((dur - t0) / spb) + 2
    for g in grids:
        k = 0
        while k * g < n_beats:
            cands.add(round(k * g, 6))
            k += 1
    scored = []
    half = max(1, int(0.045 * FPS))   # +-45ms window
    for cb in sorted(cands):
        t = t0 + cb * spb
        f = int(round(t * FPS))
        if f < 3 or f >= len(probs) - 3:
            continue
        p = probs[max(0, f - half):f + half + 1].max()
        # slight preference for coarser grid positions (cleaner rhythms)
        fine_pen = 1.0
        fr = cb % 1.0
        if min(abs(fr - r) for r in (0.0, 1.0)) < 1e-6:
            fine_pen = 1.10
        elif abs(fr - 0.5) < 1e-6:
            fine_pen = 1.05
        scored.append((p * fine_pen, cb, t, p))
    scored.sort(reverse=True)
    picked = []
    used = []
    for score, cb, t, p in scored:
        if len(picked) >= target_count:
            break
        if p < 0.06:
            continue
        if any(abs(cb - u) < min_gap_beats - 1e-6 for u in used):
            continue
        picked.append((cb, t, p))
        used.append(cb)
    picked.sort()
    return picked


def run_selection(model, vocab_toks, steps, diff_i, meter, slot, device,
                  temperature=0.9, top_p=0.95, seed=None):
    """steps: [(beat, time, prob)] -> list of (beat, row) incl hold ends."""
    rng = np.random.default_rng(seed)
    vocab = {t: i for i, t in enumerate(vocab_toks)}
    inv = vocab_toks
    n_panels = [sum(1 for c in t if c != "0") for t in inv]
    max_p = MAX_PANELS[slot]
    jump_gap = JUMP_MIN_GAP[slot]

    model.eval()
    state = None
    prev_id = len(vocab)          # BOS
    held_until = [0.0, 0.0, 0.0, 0.0]      # beat when each panel becomes free
    last_jump_t = -10.0
    out_rows = {}                 # beat -> row chars list
    hold_ends = []                # (beat, panel)

    with torch.no_grad():
        for j, (cb, t, p) in enumerate(steps):
            dtp = t - steps[j - 1][1] if j > 0 else 1.0
            dtn = steps[j + 1][1] - t if j < len(steps) - 1 else 1.0
            feats = [np.log1p(max(dtp, 0) * 10), np.log1p(max(dtn, 0) * 10)]
            feats += beat_frac_onehot(cb)
            pt = torch.tensor([[prev_id]], device=device)
            ft = torch.tensor([[feats]], dtype=torch.float32, device=device)
            d = torch.tensor([diff_i], device=device)
            m = torch.tensor([min(meter, 19)], device=device)
            logits, hold_logits, state = model(pt, ft, d, m, state)
            lg = logits[0, -1].float() / temperature

            # constraints
            for i, tok in enumerate(inv):
                if n_panels[i] > max_p:
                    lg[i] = -1e9
                elif n_panels[i] >= 2 and (t - last_jump_t) < jump_gap:
                    lg[i] = -1e9
                else:
                    for pnl in range(4):
                        if tok[pnl] != "0" and held_until[pnl] > cb + 1e-6:
                            lg[i] = -1e9
                            break
            probs = torch.softmax(lg, dim=-1).cpu().numpy()
            if probs.sum() <= 0 or not np.isfinite(probs).all():
                continue
            # top-p sample
            order = np.argsort(-probs)
            cum = np.cumsum(probs[order])
            keep = order[:max(1, int(np.searchsorted(cum, top_p) + 1))]
            pk = probs[keep] / probs[keep].sum()
            tok_id = int(rng.choice(keep, p=pk))
            tok = inv[tok_id]

            row = list(tok)
            if "2" in tok:
                hb = int(hold_logits[0, -1].argmax())
                dur = HOLD_BUCKETS[hb]
                for pnl in range(4):
                    if row[pnl] == "2":
                        end_b = cb + dur
                        # clamp: hold must end before this panel is needed again
                        held_until[pnl] = end_b
                        hold_ends.append([end_b, pnl, cb])
            if n_panels[tok_id] >= 2:
                last_jump_t = t
            out_rows.setdefault(round(cb, 6), ["0"] * 4)
            for pnl in range(4):
                if row[pnl] != "0":
                    out_rows[round(cb, 6)][pnl] = row[pnl]
            prev_id = tok_id

    # clamp hold ends so they don't cross the panel's next onset
    onset_beats = sorted(out_rows)
    panel_onsets = {p: [b for b in onset_beats
                        if out_rows[b][p] in ("1", "2")] for p in range(4)}
    rows = []
    for b in onset_beats:
        rows.append((b, "".join(out_rows[b])))
    for end_b, pnl, start_b in hold_ends:
        nxt = next((x for x in panel_onsets[pnl] if x > start_b + 1e-6), None)
        eb = end_b if (nxt is None or end_b < nxt - 0.25) else max(
            start_b + 0.5, nxt - 0.5)
        if nxt is not None and eb >= nxt:
            eb = start_b + 0.5
        eb = round(eb * 4) / 4          # keep ends on 16th grid
        row = ["0"] * 4
        row[pnl] = "3"
        rows.append((eb, "".join(row)))
    rows.sort()
    return rows


def sanitize(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "", name).strip() or "Untitled"


SPOTIFY_RE = re.compile(r"https?://open\.spotify\.com/(intl-[a-z]+/)?track/[A-Za-z0-9]+")


def resolve_spotify(url: str):
    """Spotify track URL -> (mp3 path, title, artist) via spotdl."""
    url = url.split("?")[0]
    dl_dir = os.path.join(ROOT, "data", "downloads")
    os.makedirs(dl_dir, exist_ok=True)
    print("[0/5] resolving Spotify track (spotdl)")
    # 1) metadata
    title = artist = None
    with tempfile.TemporaryDirectory() as td:
        meta_file = os.path.join(td, "meta.spotdl")
        r = subprocess.run([sys.executable, "-m", "spotdl", "save", url,
                            "--save-file", meta_file],
                           capture_output=True, text=True, timeout=120)
        if os.path.exists(meta_file):
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta:
                    title = meta[0].get("name")
                    arts = meta[0].get("artists") or []
                    artist = ", ".join(arts) if arts else meta[0].get("artist")
            except (json.JSONDecodeError, KeyError):
                pass
    # 2) audio
    before = set(glob.glob(os.path.join(dl_dir, "*.mp3")))
    r = subprocess.run([sys.executable, "-m", "spotdl", "download", url,
                        "--format", "mp3", "--output",
                        os.path.join(dl_dir, "{artists} - {title}.{output-ext}")],
                       capture_output=True, text=True, timeout=600)
    new = set(glob.glob(os.path.join(dl_dir, "*.mp3"))) - before
    if not new:
        print(r.stdout[-2000:] if r.stdout else "", file=sys.stderr)
        print(r.stderr[-2000:] if r.stderr else "", file=sys.stderr)
        sys.exit("spotdl failed to download the track")
    audio = max(new, key=os.path.getmtime)
    if not title:
        base = os.path.splitext(os.path.basename(audio))[0]
        artist, _, title = base.partition(" - ")
    print(f"      got: {artist} - {title}")
    return audio, title, artist


def main():
    ap = argparse.ArgumentParser(description="StepManAI - AI stepchart generator")
    ap.add_argument("audio", nargs="?", default=None,
                    help="audio file path or Spotify track URL (omit for form mode)")
    ap.add_argument("-t", "--title", default=None)
    ap.add_argument("-a", "--artist", default=None)
    ap.add_argument("-l", "--level", default=None,
                    help="DDR level 1-19, comma list (e.g. 6,12), or 'all'")
    ap.add_argument("--bpm", type=float, default=None, help="override detected BPM")
    ap.add_argument("--out", default=None,
                    help="pack folder to write the song into "
                         "(default: OutFox Songs\\StepManAI, or .\\output)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    # ---- form mode: prompt for anything missing --------------------------
    interactive = args.audio is None
    if interactive:
        print("=== StepManAI ===")
        while not args.audio:
            args.audio = input("Audio file or Spotify link: ").strip().strip('"')
    is_spotify = bool(SPOTIFY_RE.match(args.audio.strip()))
    if interactive:
        hint = " [auto from Spotify]" if is_spotify else ""
        if not args.title:
            args.title = input(f"Title{hint}: ").strip() or None
        if not args.artist:
            args.artist = input(f"Artist{hint}: ").strip() or None
        while not args.level:
            args.level = input("Level (1-19, or 'all'): ").strip() or None
    if not args.level:
        sys.exit("--level is required (1-19 or 'all')")

    # ---- parse levels -----------------------------------------------------
    if args.level.strip().lower() == "all":
        wanted = list(ALL_LEVELS)                      # (slot, level)
    else:
        wanted = []
        for part in args.level.split(","):
            try:
                lv = int(part)
            except ValueError:
                sys.exit(f"level must be a number 1-19 (got '{part.strip()}')")
            if not 1 <= lv <= 19:
                sys.exit(f"level must be 1-19 (got {lv})")
            wanted.append((level_to_slot(lv), lv))

    # ---- resolve audio ----------------------------------------------------
    if is_spotify:
        audio_path, sp_title, sp_artist = resolve_spotify(args.audio.strip())
        args.audio = audio_path
        args.title = args.title or sp_title
        args.artist = args.artist or sp_artist
    if not args.title:
        args.title = os.path.splitext(os.path.basename(args.audio))[0]
        print(f"      no title given, using '{args.title}'")
    if not args.artist:
        args.artist = "Unknown"
    if not os.path.exists(args.audio):
        sys.exit(f"audio not found: {args.audio}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] loading models ({device.type})")
    pl_ck = torch.load(os.path.join(CKPT, "placement.pt"), map_location=device,
                       weights_only=True)
    place = PlacementModel().to(device)
    place.load_state_dict(pl_ck["model"])
    se_ck = torch.load(os.path.join(CKPT, "selection.pt"), map_location=device,
                       weights_only=True)
    vocab_toks = se_ck["vocab"]
    select = SelectionModel(vocab=len(vocab_toks)).to(device)
    select.load_state_dict(se_ck["model"])

    print("[2/5] analysing audio")
    x = load_audio(args.audio)
    feat = features(x)
    dur = len(x) / 16000.0
    if args.bpm:
        bpm = args.bpm
        _, t0 = detect_bpm(feat)   # phase only
        # re-phase for given bpm: reuse detector phase fold
        from smai.bpm import onset_envelope
        env = onset_envelope(feat)
        period = FPS * 60.0 / bpm
        nbins = 64
        hist = np.zeros(nbins)
        for i, v in enumerate(env):
            hist[int((i % period) / period * nbins) % nbins] += v
        t0 = float((np.argmax(hist) + 0.5) / nbins * period / FPS)
        while t0 - 60.0 / bpm >= 0:
            t0 -= 60.0 / bpm
    else:
        bpm, t0 = detect_bpm(feat)
    print(f"      BPM {bpm:.2f}, first beat at {t0:.3f}s, length {dur:.0f}s")

    # sample start: strongest 15s window
    env_sec = feat[:, 1, :].astype(np.float32).sum(axis=1)
    k = 15 * FPS
    if len(env_sec) > k:
        c = np.convolve(env_sec, np.ones(k) / k, mode="valid")
        sample_start = float(np.argmax(c)) / FPS
    else:
        sample_start = 0.0

    charts = []
    for slot, level in wanted:
        grids = SLOT_GRIDS[slot]
        nps = METER_NPS.get(level, 3.0)
        target = int(nps * max(dur - t0 - 3, 10))
        print(f"[3/5] placing steps: {slot} lv{level} (~{target} steps)")
        probs = run_placement(place, feat, DIFF_IDX[slot], level, device)
        min_gap = min(g for g in grids)
        steps = pick_steps(probs, bpm, t0, grids, target, min_gap)
        print(f"      placed {len(steps)} steps")
        if len(steps) < 10:
            print("      too few steps, skipping slot")
            continue
        print(f"[4/5] choosing arrows: {slot} lv{level}")
        rows = run_selection(select, vocab_toks,
                             [(cb, t, p) for cb, t, p in steps],
                             DIFF_IDX[slot], level, slot, device,
                             temperature=args.temperature, seed=args.seed)
        charts.append((slot, level, rows))

    if not charts:
        sys.exit("no charts generated")

    print("[5/5] writing song folder")
    if args.out is None:
        args.out = OUTFOX_PACK if os.path.isdir(os.path.dirname(OUTFOX_PACK)) \
            else os.path.join(ROOT, "output")
    folder = os.path.join(args.out, sanitize(args.title))
    os.makedirs(folder, exist_ok=True)
    music_name = sanitize(args.title) + os.path.splitext(args.audio)[1].lower()
    dst_audio = os.path.join(folder, music_name)
    if os.path.abspath(args.audio) != os.path.abspath(dst_audio):
        shutil.copy2(args.audio, dst_audio)
    sm_path = os.path.join(folder, sanitize(args.title) + ".sm")
    write_sm(sm_path, args.title, args.artist, music_name, bpm,
             offset=-t0, charts=charts, sample_start=sample_start)
    print("wrote", sm_path)
    for slot, meter, rows in charts:
        n = len([r for r in rows if any(c in "12" for c in r[1])])
        print(f"  {slot:10s} meter {meter:2d}: {n} steps")


if __name__ == "__main__":
    main()
