"""Fit the lane's dash period in yards from the field's painted yard lines.

The yard lines are 5 yd apart and the start line sits on one, so every line that
crosses the far dash row is a known distance. The fitted value is YD_PER_PERIOD in
features.py; this script reproduces it and reports it per video.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from yardage import measure, yard_line_positions

ROOT = Path(__file__).resolve().parents[1]


def _one(args) -> list[tuple[str, float]]:
    rid, vid, k0 = args
    warnings.filterwarnings("ignore")
    m = measure(rid, k0)
    return [(vid, s) for _, s in yard_line_positions(m)]


def fit(s: np.ndarray) -> tuple[float, float]:
    """Scale by grid search on the share of crossings near a 5-yd line, then least squares."""
    grid = np.arange(1.90, 2.10, 0.0005)
    share = [np.mean(np.abs(s * p - 5 * np.round(s * p / 5)) < 0.25) for p in grid]
    p = grid[int(np.argmax(share))]
    target = 5 * np.round(s * p / 5)
    inl = np.abs(s * p - target) < 0.3
    return float(np.linalg.lstsq(s[inl, None], target[inl], rcond=None)[0][0]), float(max(share))


def main() -> None:
    runs = pd.read_csv(ROOT / "data/runs.csv")
    runs = runs[runs.status == "ok"]
    jobs = list(zip(runs.run_id, runs.video_id, runs.k0))
    rows = []
    with ProcessPoolExecutor(6) as ex:
        for res in ex.map(_one, jobs, chunksize=2):
            rows += res
    d = pd.DataFrame(rows, columns=["video_id", "s"])
    d = d[(d.s > 1) & (d.s < 21.5)]
    a, share = fit(d.s.values)
    print(f"all videos: {a:.4f} yd per period ({a * 0.9144:.4f} m), {share:.0%} of {len(d)} crossings within 0.25 yd")
    for vid, g in d.groupby("video_id"):
        av, sv = fit(g.s.values)
        print(f"  {vid:12s} {av:.4f}  ({sv:.0%} of {len(g)})")


if __name__ == "__main__":
    main()
