"""Parse and write StepMania .sm/.ssc simfiles.

Only dance-single charts are used. Times are absolute seconds into the audio
file (i.e. beat->time mapping honours #OFFSET, #BPMS and #STOPS).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

DIFF_ORDER = ["Beginner", "Easy", "Medium", "Hard", "Challenge", "Edit"]

# DDR-facing names accepted on the CLI -> SM difficulty slot
DDR_TO_SM = {
    "beginner": "Beginner",
    "basic": "Easy", "light": "Easy", "easy": "Easy",
    "difficult": "Medium", "standard": "Medium", "medium": "Medium", "trick": "Medium",
    "expert": "Hard", "heavy": "Hard", "hard": "Hard", "maniac": "Hard",
    "challenge": "Challenge", "oni": "Challenge",
}


@dataclass
class Note:
    time: float      # seconds into audio
    beat: float
    row: str         # 4 chars, e.g. "1001" (0 none,1 tap,2 hold start,3 hold end,4 roll,M mine)


@dataclass
class Chart:
    difficulty: str          # SM slot: Beginner/Easy/Medium/Hard/Challenge/Edit
    meter: int
    notes: list[Note] = field(default_factory=list)

    def step_rows(self) -> list[Note]:
        """Rows that contain an actual playable onset (tap/hold/roll start)."""
        return [n for n in self.notes if any(c in "124" for c in n.row)]


@dataclass
class Song:
    path: str                # song folder
    title: str = ""
    artist: str = ""
    music: str = ""          # absolute path to audio
    offset: float = 0.0
    bpms: list[tuple[float, float]] = field(default_factory=list)    # (beat, bpm)
    stops: list[tuple[float, float]] = field(default_factory=list)   # (beat, seconds)
    charts: list[Chart] = field(default_factory=list)

    # ---- beat -> seconds -------------------------------------------------
    def beat_to_time(self, beat: float) -> float:
        t = -self.offset
        bpms = self.bpms
        for i, (b, bpm) in enumerate(bpms):
            nb = bpms[i + 1][0] if i + 1 < len(bpms) else None
            if bpm <= 0:
                # negative/zero bpm (warp) - treat segment as instantaneous
                if nb is None or beat <= nb:
                    break
                continue
            seg_end = beat if (nb is None or beat < nb) else nb
            if seg_end > b:
                t += (seg_end - b) * 60.0 / bpm
            if nb is None or beat <= nb:
                break
        for sb, sdur in self.stops:
            if sb < beat:
                t += sdur
        return t


_TAG_RE = re.compile(r"#([A-Za-z]+):((?:[^;\\]|\\.)*);", re.S)


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def _parse_beat_value_list(raw: str) -> list[tuple[float, float]]:
    out = []
    for part in raw.replace("\n", "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        b, v = part.split("=", 1)
        try:
            out.append((float(b), float(v)))
        except ValueError:
            continue
    out.sort(key=lambda x: x[0])
    return out


def _parse_measures(body: str) -> list[tuple[float, str]]:
    """Return list of (beat, row) from a chart note body."""
    rows_out = []
    measures = body.split(",")
    for mi, measure in enumerate(measures):
        rows = [r.strip() for r in measure.strip().splitlines()]
        rows = [r for r in rows if r and re.fullmatch(r"[0-9A-Za-z{}|/\\*MKLF]+", r)]
        n = len(rows)
        if n == 0:
            continue
        for ri, row in enumerate(rows):
            row = row[:4]
            if len(row) < 4 or set(row) <= {"0"}:
                continue
            beat = (mi + ri / n) * 4.0
            rows_out.append((beat, row))
    return rows_out


def parse_simfile(path: str) -> Song | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    text = _strip_comments(text)
    tags = {}          # last-wins for song-level tags
    song = Song(path=os.path.dirname(path))
    is_ssc = path.lower().endswith(".ssc")

    if is_ssc:
        return _parse_ssc(text, song)

    charts = []
    for m in _TAG_RE.finditer(text):
        key, val = m.group(1).upper(), m.group(2)
        if key == "NOTES":
            charts.append(val)
        elif key not in tags:
            tags[key] = val.strip()

    song.title = tags.get("TITLE", "")
    song.artist = tags.get("ARTIST", "")
    music = tags.get("MUSIC", "")
    song.music = os.path.join(song.path, music) if music else ""
    try:
        song.offset = float(tags.get("OFFSET", "0") or 0)
    except ValueError:
        song.offset = 0.0
    song.bpms = _parse_beat_value_list(tags.get("BPMS", ""))
    song.stops = _parse_beat_value_list(tags.get("STOPS", ""))
    if not song.bpms:
        return None

    for raw in charts:
        parts = raw.split(":")
        if len(parts) < 6:
            continue
        stepstype = parts[0].strip().lower()
        if stepstype != "dance-single":
            continue
        diff = parts[2].strip().capitalize()
        if diff not in DIFF_ORDER:
            diff = "Edit"
        try:
            meter = int(float(parts[3].strip() or 0))
        except ValueError:
            meter = 0
        body = ":".join(parts[5:])
        chart = Chart(difficulty=diff, meter=meter)
        for beat, row in _parse_measures(body):
            chart.notes.append(Note(time=song.beat_to_time(beat), beat=beat, row=row))
        if chart.notes:
            song.charts.append(chart)
    return song if song.charts else None


def _parse_ssc(text: str, song: Song) -> Song | None:
    # ssc: song-level tags first, then repeated #NOTEDATA blocks with their own tags
    blocks = re.split(r"#NOTEDATA:\s*;", text)
    head = blocks[0]
    tags = {m.group(1).upper(): m.group(2).strip() for m in _TAG_RE.finditer(head)}
    song.title = tags.get("TITLE", "")
    song.artist = tags.get("ARTIST", "")
    music = tags.get("MUSIC", "")
    song.music = os.path.join(song.path, music) if music else ""
    try:
        song.offset = float(tags.get("OFFSET", "0") or 0)
    except ValueError:
        song.offset = 0.0
    song.bpms = _parse_beat_value_list(tags.get("BPMS", ""))
    song.stops = _parse_beat_value_list(tags.get("STOPS", ""))
    if not song.bpms:
        return None
    for blk in blocks[1:]:
        btags = {}
        notes_raw = None
        for m in _TAG_RE.finditer(blk):
            k = m.group(1).upper()
            if k == "NOTES":
                notes_raw = m.group(2)
            else:
                btags[k] = m.group(2).strip()
        if notes_raw is None:
            continue
        if btags.get("STEPSTYPE", "").lower() != "dance-single":
            continue
        diff = btags.get("DIFFICULTY", "Edit").capitalize()
        if diff not in DIFF_ORDER:
            diff = "Edit"
        try:
            meter = int(float(btags.get("METER", "0") or 0))
        except ValueError:
            meter = 0
        chart = Chart(difficulty=diff, meter=meter)
        for beat, row in _parse_measures(notes_raw):
            chart.notes.append(Note(time=song.beat_to_time(beat), beat=beat, row=row))
        if chart.notes:
            song.charts.append(chart)
    return song if song.charts else None


def find_simfiles(songs_root: str) -> list[str]:
    """One simfile per song folder (prefer .ssc over .sm)."""
    out = []
    for pack in os.scandir(songs_root):
        if not pack.is_dir():
            continue
        for folder in os.scandir(pack.path):
            if not folder.is_dir():
                continue
            ssc = sm = None
            for f in os.scandir(folder.path):
                fl = f.name.lower()
                if fl.endswith(".ssc"):
                    ssc = f.path
                elif fl.endswith(".sm"):
                    sm = f.path
            pick = ssc or sm
            if pick:
                out.append(pick)
    return out


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------

def rows_to_measures(notes: list[tuple[float, str]], quant: int = 48) -> str:
    """notes: (beat, row). quant = rows per beat grid unit denominator base.

    Snaps each note to 1/quant beats, groups into measures, and picks the
    smallest per-measure subdivision that represents all rows exactly.
    """
    if not notes:
        return "0000\n0000\n0000\n0000"
    grid: dict[int, str] = {}
    for beat, row in notes:
        tick = round(beat * quant)
        if tick in grid:
            grid[tick] = _merge_rows(grid[tick], row)
        else:
            grid[tick] = row
    last_measure = max(grid) // (quant * 4)
    measures = []
    for mi in range(last_measure + 1):
        m_ticks = {t - mi * quant * 4: r for t, r in grid.items()
                   if mi * quant * 4 <= t < (mi + 1) * quant * 4}
        # choose subdivision: rows per measure among common values
        for rpm in (4, 8, 12, 16, 24, 32, 48, 64, 96, 192):
            step = quant * 4 // rpm
            if quant * 4 % rpm == 0 and all(t % step == 0 for t in m_ticks):
                break
        lines = []
        for i in range(rpm):
            lines.append(m_ticks.get(i * step, "0000"))
        measures.append("\n".join(lines))
    return "\n,\n".join(measures)


def _merge_rows(a: str, b: str) -> str:
    return "".join(y if x == "0" else x for x, y in zip(a, b))


def write_sm(path: str, title: str, artist: str, music_file: str, bpm: float,
             offset: float, charts: list[tuple[str, int, list[tuple[float, str]]]],
             sample_start: float = 30.0, banner: str = "", background: str = "",
             credit: str = "StepManAI") -> None:
    """charts: list of (sm_difficulty, meter, [(beat,row)...])"""
    lines = [
        f"#TITLE:{title};",
        f"#ARTIST:{artist};",
        f"#CREDIT:{credit};",
        f"#BANNER:{banner};",
        f"#BACKGROUND:{background};",
        f"#MUSIC:{music_file};",
        f"#OFFSET:{offset:.3f};",
        f"#SAMPLESTART:{sample_start:.3f};",
        "#SAMPLELENGTH:15;",
        f"#DISPLAYBPM:{bpm:.0f};",
        f"#BPMS:0={bpm:.3f};",
        "#STOPS:;",
    ]
    for diff, meter, notes in charts:
        lines += [
            "",
            f"//---------------dance-single - {credit}----------------",
            "#NOTES:",
            "     dance-single:",
            f"     {credit}:",
            f"     {diff}:",
            f"     {meter}:",
            "     0,0,0,0,0:",
            rows_to_measures(notes),
            ";",
        ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
