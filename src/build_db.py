"""Assemble the dataset: runs table with QC and split, per-frame series, and the\nclock window resampled on a common time grid and on a common distance grid."""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from features import series
from qc import logo_reference, metrics, verdict, window
from yardage import load, measure

ROOT = Path(__file__).resolve().parents[1]
CLASS = {"skill": "SKILL", "strong": "STRONG", "line": "LINEMAN"}
GROUP = {"WR": "WR", "CB": "DB", "SAF": "DB", "RB": "RB", "TE": "TE", "LB": "LB",
         "EDGE": "DL", "DT": "DL", "OT|G|C": "OL"}
POSITION = {"OT|G|C": "OL"}
GRID = np.round(np.arange(0, 40.0001, 0.25), 2)
SEED = 2026
PHASES = 101


def _logo_refs(runs: pd.DataFrame) -> dict[str, np.ndarray]:
    refs = {}
    for vid, g in runs.groupby("video_id"):
        sample = [load(r) for r in g.run_id[:6]]
        refs[vid] = logo_reference(sample)
    return refs


def _one(args) -> tuple[str, dict, pd.DataFrame | None]:
    run, logo_ref = args
    warnings.filterwarnings("ignore")
    try:
        m = measure(run.run_id, run.k0)
        df = series(m, run.k0)
        mt = metrics(run, m, df, logo_ref)
    except Exception as e:  # a broken cache or degenerate run is a QC failure, not a crash
        return run.run_id, {"error": repr(e)}, None
    return run.run_id, mt, df


def split_athletes(ath: pd.DataFrame) -> dict[str, str]:
    """Alternate athletes into train/test within each position, from a seeded shuffle."""
    rng = np.random.default_rng(SEED)
    out = {}
    for cls, g in ath.groupby("cls"):
        flip = 0
        for pos, gp in g.groupby("position"):
            keys = list(gp.athlete)
            rng.shuffle(keys)
            for key in keys:
                out[key] = "train" if flip % 2 == 0 else "test"
                flip += 1
    return out


def _resample(src: pd.DataFrame, t_at: np.ndarray, c: str) -> np.ndarray:
    s = src[["t_clock", c]].dropna()
    if len(s) < 2:
        return np.full(len(t_at), np.nan)
    if c == "foot_contact":
        idx = np.clip(np.searchsorted(s.t_clock.values, t_at), 1, len(s) - 1)
        near = np.where(t_at - s.t_clock.values[idx - 1] < s.t_clock.values[idx] - t_at, idx - 1, idx)
        return s[c].values[near]
    # No extrapolation: before the first stride the step channels stay empty.
    return np.round(np.interp(t_at, s.t_clock, s[c], left=np.nan, right=np.nan), 3)


def by_distance(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    live = df[df.x_yd.notna()]
    x = np.maximum.accumulate(live.x_yd.values)
    t = live.t_clock.values
    # Distance must strictly increase to be inverted.
    keep = np.concatenate([[True], np.diff(x) > 1e-6])
    x, t = x[keep], t[keep]
    grid = GRID[(GRID >= x.min()) & (GRID <= x.max())]
    t_at = np.interp(grid, x, t)
    out = {"x_yd": grid, "t_clock": np.round(t_at, 4)}
    for c in channels:
        out[c] = _resample(df, t_at, c)
    return pd.DataFrame(out)


def by_time(full: pd.DataFrame, final: float, channels: list[str]) -> pd.DataFrame:
    """The clock window resampled to PHASES points, clock start (0) to clock stop (1)."""
    t_at = np.linspace(0, final, PHASES)
    # One frame either side lets the end points interpolate rather than clamp.
    near = full[(full.t_clock >= -1 / 29.97 - 1e-6) & (full.t_clock <= final + 1 / 29.97 + 1e-6)]
    out = {"phase": np.round(np.linspace(0, 1, PHASES), 3), "t_clock": np.round(t_at, 4)}
    for c in ["x_yd"] + channels:
        out[c] = _resample(near, t_at, c)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    videos = pd.read_csv(ROOT / "data/videos.csv")
    runs = pd.read_csv(ROOT / "data/runs_raw.csv").merge(videos[["video_id", "cls"]], on="video_id")
    runs["cls"] = runs.cls.map(CLASS)
    runs["position"] = runs.positions.map(lambda p: POSITION.get(p, p))
    # The panel's own position code: a group session can include other positions (a QB
    # runs with the WRs), which are not part of any class.
    runs = runs.merge(pd.read_csv(ROOT / "data/panel_positions.csv")[["run_id", "panel_pos"]],
                      on="run_id", how="left")
    runs["athlete"] = runs.panel_pos + "-" + runs.bib.astype("Int64").astype(str)
    runs["off_group"] = runs.panel_pos != runs.positions.map(GROUP)
    cached = runs.run_id.map(lambda r: (ROOT / "cache/frames" / f"{r}.pkl").exists())
    print(f"{cached.sum()}/{len(runs)} runs extracted")
    runs = runs[cached].copy()

    refs = _logo_refs(runs)
    jobs = [(row, refs[row.video_id]) for _, row in runs.iterrows()]
    results = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for rid, mt, df in ex.map(_one, jobs, chunksize=2):
            results[rid] = (mt, df)

    rows = []
    for r in runs.itertuples():
        mt, _ = results[r.run_id]
        ok, why = verdict(mt) if "error" not in mt else (False, mt["error"])
        rows.append({"run_id": r.run_id, **{k: v for k, v in mt.items() if k != "error"},
                     "qc_pass": ok, "qc_reason": why})
    q = pd.DataFrame(rows)
    runs = runs.drop(columns=[c for c in q.columns if c != "run_id" and c in runs.columns]).merge(q, on="run_id")

    # Consecutive group uploads overlap at the seam: same athlete, same time.
    order = {v: i for i, v in enumerate(videos.video_id)}
    runs["_v"] = runs.video_id.map(order)
    runs = runs.sort_values(["_v", "t_origin"])
    # Only across videos: one athlete can post the same time on both attempts.
    first_video = runs.groupby(["athlete", "final_time"]).video_id.transform("first")
    runs["duplicate"] = runs.video_id != first_video
    runs["attempt"] = runs[~runs.duplicate & ~runs.off_group].groupby("athlete").cumcount() + 1
    runs["status"] = np.select([runs.off_group, runs.duplicate, runs.qc_pass],
                               ["excluded", "duplicate", "ok"], "rejected")
    runs.loc[runs.off_group, "qc_reason"] = "panel position " + runs.panel_pos.astype(str)

    good = runs[runs.status == "ok"]
    ath = good.groupby("athlete").agg(cls=("cls", "first"), position=("position", "first")).reset_index()
    split = split_athletes(ath)
    runs["split"] = runs.athlete.map(split)

    series_rows, dist_rows, time_rows = [], [], []
    channels = None
    for r in runs[runs.status == "ok"].itertuples():
        full = results[r.run_id][1]
        df = full[full.frame.isin(window(r.k0, r.final_time, len(full)))].copy()
        channels = [c for c in df.columns if c not in ("frame", "t_clock", "x_yd", "valid")]
        dist = by_distance(df, channels)
        tgrid = by_time(full, r.final_time, channels)
        for d in (df, dist, tgrid):
            d.insert(0, "run_id", r.run_id)
        series_rows.append(df)
        dist_rows.append(dist)
        time_rows.append(tgrid)
    empty = pd.DataFrame(columns=["run_id", "x_yd", "t_clock"])
    ser = pd.concat(series_rows, ignore_index=True) if series_rows else empty
    dist = pd.concat(dist_rows, ignore_index=True) if dist_rows else empty
    tser = pd.concat(time_rows, ignore_index=True) if time_rows else empty

    # Held-out check: panel 10-yd split vs the time the measured hip crosses 10 yd.
    cross = dist[dist.x_yd == 10.0].set_index("run_id").t_clock
    runs["t_at_10yd"] = runs.run_id.map(cross)
    runs["t_at_40yd"] = runs.run_id.map(dist[dist.x_yd == 40.0].set_index("run_id").t_clock)

    cols = ["run_id", "video_id", "group", "position", "panel_pos", "cls", "athlete", "bib", "attempt", "split",
            "status", "qc_reason", "final_time", "split10", "t_at_10yd", "t_at_40yd",
            "clip_ss", "clip_frames", "k0", "clock_resid_p95", "clock_coverage", "bib_agree",
            "panel_min_corr", "max_cut", "turf_min", "turf_median", "lane_frac", "track_frac", "start_support",
            "x_at_zero", "x_at_stop", "max_step_yd"]
    runs = runs.sort_values("run_id")[cols]
    for c in runs.select_dtypes("float").columns:
        runs[c] = runs[c].round(4)
    runs.to_csv(ROOT / "data/runs.csv", index=False)
    ser.to_parquet(ROOT / "data/series.parquet", index=False)
    ser.to_csv(ROOT / "data/series.csv", index=False)
    dist.to_parquet(ROOT / "data/series_by_distance.parquet", index=False)
    tser.to_parquet(ROOT / "data/series_by_time.parquet", index=False)

    ok = runs[runs.status == "ok"]
    print(runs.status.value_counts().to_string())
    print(ok.groupby(["cls", "split"]).agg(runs=("run_id", "size"), athletes=("athlete", "nunique")).to_string())
    print(f"series: {len(ser)} rows, {ser.run_id.nunique()} runs; by time: {len(tser)} rows; "
          f"by distance: {len(dist)} rows")


if __name__ == "__main__":
    main()
