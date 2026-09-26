"""Find every 40-yd run in each video from the panel clock.

Coarse pass: read the panel at 4 fps; a run is a stretch where one field counts up
at real-time rate and then freezes on a 40 time. Fine pass: read that field on every
frame of the run and fit t = (frame - f0) / fps, so each frame gets a clock time.

Writes cache/scan/<video>.csv and data/runs_raw.csv.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from panel import BAND, Reader, frames

ROOT = Path(__file__).resolve().parents[1]
FPS = 30000 / 1001
SCAN_FPS = 4
PRE, POST = 1.0, 0.6


def scan(video: Path, reader: Reader) -> pd.DataFrame:
    rows = []
    for i, band in frames(video, fps=SCAN_FPS, crop=BAND):
        t = i / SCAN_FPS
        for xc, v in reader.times(band):
            rows.append((t, xc, v, reader.bib(band)))
    return pd.DataFrame(rows, columns=["t", "x", "value", "bib"])


def _slots(x: pd.Series, tol: float = 30) -> pd.Series:
    """Label each field position by its cluster's median x. A field's centre jitters by
    a few pixels with its digits (a "1" is narrower), so fixed bins would split it."""
    xs = np.sort(x.unique())
    groups = np.split(xs, np.nonzero(np.diff(xs) > tol)[0] + 1)
    label = {v: int(np.median(g)) for g in groups for v in g}
    return x.map(label)


def _field(fields: list[tuple[int, float | None]], x0: float | None, tol: float = 30) -> float | None:
    near = [v for x, v in fields if x0 is not None and abs(x - x0) < tol]
    return near[0] if near else None


def find_runs(sc: pd.DataFrame) -> list[dict]:
    """Counting stretches per slot that freeze on a plausible 40 time."""
    sc = sc.dropna(subset=["value"]).copy()
    sc["slot"] = _slots(sc.x)
    runs = []
    for slot, g in sc.groupby("slot"):
        g = g.sort_values("t").reset_index(drop=True)
        origin = g.t - g.value
        dt, dv = g.t.diff(), g.value.diff()
        counting = (g.value > 0) & (dt < 0.3) & ((dv - dt).abs() < 0.08)
        i, n = 0, len(g)
        while i < n:
            if not counting[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and counting[j + 1] and abs(origin[j + 1] - origin[i]) < 0.1:
                j += 1
            # The stop is the first value that repeats once counting ends.
            after = g[(g.t > g.t[j]) & (g.t <= g.t[j] + 1.5)].value.values
            rep = [a for a, b in zip(after[:-1], after[1:]) if a == b]
            if j - i >= 3 and rep:
                runs.append(dict(slot=slot, x=int(g.x[i:j + 1].median()),
                                 t_origin=float(np.median(origin[i:j + 1])),
                                 final=float(rep[0]), t_last=float(g.t[j])))
            i = j + 1
    dashes = [r for r in runs if 3.9 <= r["final"] <= 6.5]
    for d in dashes:
        # The 10-yd split does not count live: its slot holds 0.00, then shows the split.
        w = sc[(sc.slot != d["slot"]) & (sc.t > d["t_origin"]) & (sc.t < d["t_origin"] + d["final"] + 1.0)
               & sc.value.between(1.2, 2.5)]
        m = w.groupby("slot").value.agg(lambda v: v.mode().iloc[0] if len(v) >= 3 else None).dropna()
        d["split10_coarse"] = float(m.iloc[0]) if len(m) else None
        d["split_x"] = int(w[w.slot == m.index[0]].x.median()) if len(m) else None
    return sorted(dashes, key=lambda r: r["t_origin"])


def fine(video: Path, reader: Reader, run: dict) -> dict:
    """Frame-accurate clock over the run window."""
    f_start = int(np.floor((run["t_origin"] - PRE) * FPS))
    ss = f_start / FPS
    dur = PRE + run["final"] + POST
    k, val, split, bibs = [], [], [], []
    for i, band in frames(video, ss=ss, dur=dur, crop=BAND):
        fields = reader.times(band)
        k.append(i)
        val.append(_field(fields, run["x"]))
        split.append(_field(fields, run["split_x"]))
        bibs.append(reader.bib(band))
    k = np.array(k)
    v = np.array([np.nan if x is None else x for x in val])
    counting = (v > 0) & (v < run["final"] - 1e-6)
    kk, vv = k[counting], v[counting]
    # Slope is fixed by the frame rate; only the origin is free. Median is robust to misreads.
    k0 = float(np.median(kk - vv * FPS)) if len(kk) else np.nan
    resid = vv - (kk - k0) / FPS
    frozen = v[k > k0 + run["final"] * FPS]
    final = Counter(frozen[~np.isnan(frozen)]).most_common(1)
    s = np.array([np.nan if x is None else x for x in split])
    s_final = Counter(s[(k > k0 + 2.5 * FPS) & ~np.isnan(s)]).most_common(1)
    bib = Counter(b for b in bibs if b).most_common(1)
    return dict(
        clip_ss=round(ss, 4), clip_frames=len(k), k0=round(k0, 3),
        final_time=final[0][0] if final else np.nan,
        split10=s_final[0][0] if s_final else np.nan,
        clock_resid_p95=round(float(np.percentile(np.abs(resid), 95)), 4) if len(resid) else np.nan,
        clock_monotone=bool(np.all(np.diff(vv) >= -1e-6)),
        n_clock=int(len(kk)),
        bib=bib[0][0] if bib else None,
        bib_agree=round(bib[0][1] / max(1, sum(1 for b in bibs if b)), 3) if bib else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    # Comma-separated: some ids start with "-" and would parse as flags.
    ap.add_argument("--videos", default="")
    ap.add_argument("--out", type=Path, default=ROOT / "data/runs_raw.csv")
    ap.add_argument("--scan-dir", type=Path, default=ROOT / "cache/scan")
    a = ap.parse_args()
    videos = pd.read_csv(ROOT / "data/videos.csv")
    if a.videos:
        videos = videos[videos.video_id.isin(a.videos.split(","))]
    reader = Reader()
    a.scan_dir.mkdir(parents=True, exist_ok=True)
    out_path = a.out
    prev = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    rows = []
    for v in videos.itertuples():
        path = ROOT / "video" / f"{v.video_id}.mp4"
        if not path.exists():
            print(v.video_id, "missing video")
            continue
        scan_path = a.scan_dir / f"{v.video_id}.csv"
        sc = pd.read_csv(scan_path) if scan_path.exists() else scan(path, reader)
        sc.to_csv(scan_path, index=False)
        runs = find_runs(sc)
        print(v.video_id, len(runs), "runs", flush=True)
        for n, r in enumerate(runs):
            f = fine(path, reader, r)
            rows.append(dict(run_id=f"{v.video_id}_{n:03d}", video_id=v.video_id, group=v.group,
                             positions=v.positions, **{k: r[k] for k in ("slot", "t_origin")},
                             split10_coarse=r["split10_coarse"], **f))
    df = pd.DataFrame(rows)
    if len(prev):
        df = pd.concat([prev[~prev.video_id.isin(df.video_id.unique())], df])
    df.sort_values("run_id").to_csv(out_path, index=False)
    print(f"{len(df)} runs -> {out_path}")


if __name__ == "__main__":
    main()
