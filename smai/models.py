"""Neural models: step placement (CNN+BiGRU) and step selection (LSTM).

Sized for a GTX 1650 (4GB, shared with desktop) — small but real.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_DIFF = 6          # Beginner Easy Medium Hard Challenge Edit
MAX_METER = 20


class PlacementModel(nn.Module):
    """Per-frame step-onset probability from (T,3,80) mels + difficulty."""

    def __init__(self, rnn_hidden: int = 128, rnn_layers: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(7, 3), padding=(3, 1)),   # (time, freq)
            nn.ReLU(),
            nn.MaxPool2d((1, 3)),
            nn.Conv2d(24, 48, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((1, 3)),
        )
        conv_out = 48 * (80 // 3 // 3)      # 48 * 8 = 384
        self.diff_emb = nn.Embedding(N_DIFF, 16)
        self.meter_emb = nn.Embedding(MAX_METER, 16)
        self.rnn = nn.GRU(conv_out + 32, rnn_hidden, num_layers=rnn_layers,
                          batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, mel: torch.Tensor, diff: torch.Tensor, meter: torch.Tensor):
        # mel: (B, T, 3, 80); diff/meter: (B,)
        x = mel.permute(0, 2, 1, 3)                # (B, 3, T, 80)
        x = self.conv(x)                           # (B, 48, T, 8)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        cond = torch.cat([self.diff_emb(diff), self.meter_emb(meter)], dim=-1)
        x = torch.cat([x, cond.unsqueeze(1).expand(-1, t, -1)], dim=-1)
        x, _ = self.rnn(x)
        return self.head(x).squeeze(-1)            # (B, T) logits


class SelectionModel(nn.Module):
    """Autoregressive next-step-token model.

    Token = 4 panels each in {0 none, 1 tap, 2 hold-start}; vocab built from
    data. Conditioned on difficulty, meter, time-delta and beat-fraction of
    the *current* step being predicted (rhythmic context).
    """

    def __init__(self, vocab: int, hidden: int = 256, layers: int = 2,
                 n_hold_buckets: int = 8):
        super().__init__()
        self.vocab = vocab
        self.tok_emb = nn.Embedding(vocab + 1, 64)          # +1 = BOS
        self.diff_emb = nn.Embedding(N_DIFF, 12)
        self.meter_emb = nn.Embedding(MAX_METER, 12)
        # numeric features: log dt_prev, log dt_next, beat_frac one-hot(5)
        self.num_proj = nn.Linear(2 + 5, 32)
        self.rnn = nn.LSTM(64 + 24 + 32, hidden, num_layers=layers,
                           batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, vocab)
        self.hold_head = nn.Linear(hidden, n_hold_buckets)   # duration buckets

    def forward(self, prev_tokens, num_feats, diff, meter, state=None):
        # prev_tokens: (B, L) ids of previous step (BOS=vocab)
        # num_feats: (B, L, 7)
        b, l = prev_tokens.shape
        cond = torch.cat([self.diff_emb(diff), self.meter_emb(meter)], -1)
        x = torch.cat([
            self.tok_emb(prev_tokens),
            cond.unsqueeze(1).expand(-1, l, -1),
            self.num_proj(num_feats),
        ], dim=-1)
        out, state = self.rnn(x, state)
        return self.head(out), self.hold_head(out), state


# hold duration buckets in beats
HOLD_BUCKETS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def hold_bucket(beats: float) -> int:
    best, bi = 1e9, 0
    for i, b in enumerate(HOLD_BUCKETS):
        d = abs(beats - b)
        if d < best:
            best, bi = d, i
    return bi


def beat_frac_onehot(beat: float) -> list[float]:
    """Classify a beat position: on-beat, 8th, 16th, 12th/24th, other."""
    f = beat % 1.0
    def near(x, g):
        return min(abs(x - r) for r in g) < 1e-3
    if near(f, [0.0, 1.0]):
        return [1, 0, 0, 0, 0]
    if near(f, [0.5]):
        return [0, 1, 0, 0, 0]
    if near(f, [0.25, 0.75]):
        return [0, 0, 1, 0, 0]
    if near(f, [1/3, 2/3, 1/6, 5/6]):
        return [0, 0, 0, 1, 0]
    return [0, 0, 0, 0, 1]
