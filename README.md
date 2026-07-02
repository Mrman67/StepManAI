# StepManAI

AI stepchart generator for StepMania / Project OutFox, trained on your own
Songs library (~1800 official DDR + ITG simfiles). Give it an audio file or a
Spotify link plus a DDR level (1-19) — it writes a ready-to-play song folder.

**Colab (recommended):**
- Generate: [colab/stepmanai_generate.ipynb](https://colab.research.google.com/github/Mrman67/StepManAI/blob/main/colab/stepmanai_generate.ipynb)
- Train: [colab/stepmanai_train.ipynb](https://colab.research.google.com/github/Mrman67/StepManAI/blob/main/colab/stepmanai_train.ipynb)
  (needs `stepmanai_dataset.tar` in `Drive/StepManAI/`, built locally with
  `scan_library.py` + `build_cache.py` + `pack_dataset.py`)

Architecture (Dance Dance Convolution-style, two-stage):

1. **Placement model** — CNN + bidirectional GRU over multi-scale log-mel
   spectrograms (3 FFT sizes x 80 mels @ 100 fps), conditioned on difficulty
   slot + meter. Predicts *when* steps happen.
2. **Selection model** — 2-layer LSTM over step tokens, conditioned on
   difficulty, time-deltas and beat fraction. Predicts *which* arrows
   (taps, jumps, freezes + freeze lengths).

Generation: detect BPM + beat phase → placement probabilities → snap peaks to
the beat grid allowed for that difficulty → take the strongest N (N from the
meter→density curve measured on your library) → sample arrows autoregressively
with playability constraints (no taps on held panels, jump spacing, max 2
panels).

## Usage

Form mode (just run it and answer the prompts):

```
> python generate.py
Audio file or Spotify link: https://open.spotify.com/track/XXXX
Title  [auto from Spotify]:
Artist [auto from Spotify]:
Level (1-19, or 'all'):    12
```

Non-interactive:

```
python generate.py song.mp3 -t "Song Name" -a "Artist" -l 12
python generate.py https://open.spotify.com/track/XXXX -l all
python generate.py song.ogg -t N -a A -l 6,12,15 --bpm 174
```

Level is the DDR difficulty number **1-19** (`all` = a 5-chart song at
3/6/9/12/15). Spotify links auto-fill title/artist and download the audio via
spotdl. Output goes to `C:\Games\OutFox 0.5.0 Alpha Win64\Songs\StepManAI\<Title>\`
by default (`--out` to change). Reload songs in OutFox afterwards.

Useful knobs: `--seed N` (reproducible arrows), `--temperature 0.8`
(tamer patterns) / `1.1` (spicier), `--bpm` (if detection picks a wrong
tempo/half-tempo).

## Retraining

```
python scan_library.py       # index Songs -> data/index.json
python build_cache.py        # mel features -> data/cache (~9 GB)
python train_placement.py    # -> checkpoints/placement.pt   (GPU, ~1-2 h)
python train_selection.py    # -> checkpoints/selection.pt   (~20 min)
```

Files: `smai/simfile.py` (.sm/.ssc parser + writer), `smai/audio.py`
(features), `smai/bpm.py` (tempo detection), `smai/models.py` (both nets).

Limitations: constant-BPM output (no BPM changes/stops), no mines/shock
arrows, dance-single only.
