# StepManAI

AI stepchart generator for StepMania / Project OutFox, trained on ~1800
official DDR arcade simfiles. Give it an audio file or a Spotify link plus a
DDR level (1-19) — it writes a ready-to-play song folder.

Everything runs on Google Colab — nothing to install locally.

## 1. Train (once)

Open [colab/stepmanai_train.ipynb](https://colab.research.google.com/github/Mrman67/StepManAI/blob/main/colab/stepmanai_train.ipynb),
set the runtime to **T4 GPU**, and run the cells top to bottom.

The first run downloads the 19 official DDR arcade packs from
Zenius-I-Vanisher, builds the audio-feature dataset on the Colab disk, and
saves a reusable `stepmanai_dataset.tar` to your Google Drive. Training then
takes ~1 hour; checkpoints land in `Drive/StepManAI/checkpoints/`.

## 2. Generate

Open [colab/stepmanai_generate.ipynb](https://colab.research.google.com/github/Mrman67/StepManAI/blob/main/colab/stepmanai_generate.ipynb)
and fill in the form:

- **spotify_link** — paste a Spotify track URL (title/artist auto-fill), or
  leave empty to upload an audio file
- **level** — DDR difficulty number 1-19, or tick **all_difficulties** for a
  full 5-chart song (levels 3/6/9/12/15)
- **temperature** — 0.8 tamer patterns, 1.1 spicier; **seed** — reproducible arrows

Run the cell; it downloads a `.zip`. Unzip into your `Songs\StepManAI\` folder
and reload songs in OutFox/StepMania.

## Architecture

Dance Dance Convolution-style, two stages:

1. **Placement model** — CNN + bidirectional GRU over multi-scale log-mel
   spectrograms (3 FFT sizes x 80 mels @ 100 fps), conditioned on difficulty
   slot + meter. Predicts *when* steps happen.
2. **Selection model** — 2-layer LSTM over step tokens, conditioned on
   difficulty, time-deltas and beat fraction. Predicts *which* arrows
   (taps, jumps, freezes + freeze lengths).

Generation: detect BPM + beat phase → placement probabilities → snap peaks to
the beat grid allowed for that difficulty → take the strongest N (N from the
meter→density curve measured on the training library) → sample arrows
autoregressively with playability constraints (no taps on held panels, jump
spacing, max 2 panels).

Files: `smai/simfile.py` (.sm/.ssc parser + writer), `smai/audio.py`
(features), `smai/bpm.py` (tempo detection), `smai/models.py` (both nets),
`generate.py` (CLI used by the generate notebook — also runs locally if you
ever want that).

Limitations: constant-BPM output (no BPM changes/stops), no mines/shock
arrows, dance-single only.
