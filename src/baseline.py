"""Sanity check that the series separate the classes: 1-nearest-neighbour on an
equal-length view (time grid by default, or --grid distance), train split -> test
split, per channel group. Not a model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "progress (t_clock / x_yd)": ["t_clock", "x_yd"],
    "kinematics (v, a)": ["v_yds", "a_yds2"],
    "posture (trunk, hip height)": ["trunk_angle", "hip_height_ratio"],
    "joint angles": ["knee_lead", "knee_trail", "hip_flex_lead", "hip_flex_trail",
                     "shin_angle_lead", "thigh_sep", "elbow_angle"],
    "stride (freq, length)": ["step_freq_hz", "step_len_yd"],
    "pose only (all pose channels)": ["trunk_angle", "hip_height_ratio", "knee_lead", "knee_trail",
                                      "hip_flex_lead", "hip_flex_trail", "shin_angle_lead",
                                      "thigh_sep", "elbow_angle", "arm_swing"],
}


def tensor(data: pd.DataFrame, runs: list[str], cols: list[str], key: str, grid: np.ndarray) -> np.ndarray:
    out = np.full((len(runs), len(grid), len(cols)), np.nan)
    for rid, g in data[data.run_id.isin(runs)].groupby("run_id", sort=False):
        j = runs.index(rid)
        pos = np.searchsorted(grid, g[key].values)
        out[j, pos] = g[cols].values
    # Fill gaps along the grid, then z-score each channel with train statistics later.
    for i in range(len(runs)):
        for c in range(len(cols)):
            x = out[i, :, c]
            ok = ~np.isnan(x)
            out[i, :, c] = np.interp(np.arange(len(x)), np.nonzero(ok)[0], x[ok]) if ok.sum() > 1 else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["time", "distance"], default="time")
    a = ap.parse_args()
    runs = pd.read_csv(ROOT / "data/runs.csv")
    runs = runs[runs.status == "ok"]
    if a.grid == "time":
        data = pd.read_parquet(ROOT / "data/series_by_time.parquet")
        key, grid = "phase", np.round(np.linspace(0, 1, 101), 3)
    else:
        data = pd.read_parquet(ROOT / "data/series_by_distance.parquet")
        key, grid = "x_yd", np.round(np.arange(0, 39.0001, 0.25), 2)
    tr, te = runs[runs.split == "train"], runs[runs.split == "test"]
    print(f"{a.grid} grid: train {len(tr)} runs / test {len(te)} runs")
    for name, cols in GROUPS.items():
        cols = [c for c in cols if c != key]
        Xtr = tensor(data, list(tr.run_id), cols, key, grid)
        Xte = tensor(data, list(te.run_id), cols, key, grid)
        mu = Xtr.mean((0, 1))
        sd = Xtr.std((0, 1)) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        d = ((Xte[:, None] - Xtr[None]) ** 2).sum((2, 3))
        pred = tr.cls.values[d.argmin(1)]
        acc = (pred == te.cls.values).mean()
        print(f"{name:32s} 1-NN accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
