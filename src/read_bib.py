"""Identify the athlete on screen from the panel's bib number.

Combine bibs are assigned alphabetically by last name within a position group,
covering every invitee in that group including those who never post a 40. Reading
the two-digit bib and indexing the roster is therefore cheaper and less error-prone
than OCRing the name, which would need a full alphabet of prototypes.
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

from read_clock import PanelReader, band_frames, BAND, _norm

COMBINE_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
               "combine/combine.parquet")
SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
NAME_X = (200, 700)
# The bib renders smaller than the timing digits, so the same prototypes correlate
# lower; a single frame is not trusted on its own, see bib_for_window.
BIB_MIN_CORR = 0.70


def sort_key(name: str) -> tuple[str, str]:
    parts = [p for p in name.split() if p.lower().strip(".") not in
             {s.strip(".") for s in SUFFIXES}]
    last = parts[-1] if parts else name
    first = parts[0] if parts else ""
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    return norm(last), norm(first)


def roster(season: int, positions: list[str]) -> pd.DataFrame:
    c = pd.read_parquet(COMBINE_URL)
    d = c[(c.season == season) & (c.pos.isin(positions))].copy()
    d = d.sort_values("player_name", key=lambda s: s.map(sort_key))
    d = d.reset_index(drop=True)
    d["bib"] = d.index + 1
    return d[["bib", "player_name", "pos", "forty"]]


def read_bib(band: np.ndarray, reader: PanelReader) -> int | None:
    """Return the bib number from the cyan line under the athlete's name."""
    r, g, b = (band[..., i].astype(int) for i in range(3))
    cyan = (b > 170) & (g > 150) & (r < 160) & (b > r + 40)
    lab, _ = ndimage.label(cyan)

    # Confine to the name block. Cyan also appears in the leaderboard text at the
    # right of the panel, which otherwise supplies the trailing group.
    marks = []
    for sl in ndimage.find_objects(lab):
        x, y = sl[1].start, sl[0].start
        h = sl[0].stop - y
        w = sl[1].stop - x
        if 20 < h < 40 and 3 < w < 40 and NAME_X[0] <= x <= NAME_X[1]:
            marks.append((x, w, y, _norm(cyan[sl])))
    if len(marks) < 3:
        return None
    # Keep a single text line.
    ys = np.array([m[2] for m in marks])
    marks = [m for m in marks if abs(m[2] - np.median(ys)) < 8]
    marks.sort()

    # Prefix letters sit tight together; a wider gap separates them from the number.
    groups, cur = [], [marks[0]]
    for m in marks[1:]:
        if m[0] - (cur[-1][0] + cur[-1][1]) > 15:
            groups.append(cur)
            cur = []
        cur.append(m)
    groups.append(cur)
    if len(groups) < 2:
        return None

    digits, confs = zip(*(reader._digit(m[3]) for m in groups[-1]))
    if min(confs) < BIB_MIN_CORR or not all(d.isdigit() for d in digits):
        return None
    return int("".join(digits))


def bib_for_window(video: str, t0: float, t1: float, reader: PanelReader,
                   n: int = 8) -> tuple[int | None, float]:
    """Majority-vote the bib across a run window; returns (bib, agreement)."""
    votes = []
    for k in range(n):
        t = t0 + (t1 - t0) * k / max(n - 1, 1)
        fr = band_frames(video, t, 1.0, "1")
        if not fr:
            continue
        b = read_bib(fr[0], reader)
        if b is not None:
            votes.append(b)
    if not votes:
        return None, 0.0
    vals, counts = np.unique(votes, return_counts=True)
    i = int(np.argmax(counts))
    return int(vals[i]), float(counts[i] / len(votes))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--positions", nargs="+", required=True)
    ap.add_argument("--at", type=float, nargs="+", required=True)
    ap.add_argument("--templates", default="data/glyph_templates.npz")
    args = ap.parse_args()

    ros = roster(args.season, args.positions)
    reader = PanelReader(args.templates)
    by_bib = dict(zip(ros.bib, ros.player_name))

    for t in args.at:
        fr = band_frames(args.video, t, 1.0, "1")
        bib = read_bib(fr[0], reader) if fr else None
        name = by_bib.get(bib, "?") if bib else "-"
        print(f"  t={t:7.1f}s  bib={str(bib):>4}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
