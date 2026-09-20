"""Turn the hand labels into the study's position-time series.

Two outputs: the measured crossings themselves, and a model series resampled on
a fixed grid. The fit uses only the labels, so the official 40 time stays an
independent check on them.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

MIN_MARKS = 4


def model(t, vmax, tau):
    return vmax * (t + tau * np.exp(-t / tau) - tau)


def fit(t, x):
    """v_max, tau and the motion onset relative to the timer.

    The clock starts on the start sensor, by which point the athlete is already
    accelerating; holding onset at zero forces the curve through a point the
    sprint was never at and doubles the residuals.
    """
    s = least_squares(lambda p: model(np.maximum(t - p[2], 0), p[0], p[1]) - x,
                      [9.0, 0.9, 0.0], bounds=([5, 0.2, -0.4], [14, 3.0, 0.4]))
    vmax, tau, t0 = s.x
    return vmax, tau, t0, float(np.sqrt(np.mean(s.fun ** 2))), float(np.max(np.abs(s.fun)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--plan", default="data/label_plan.csv")
    ap.add_argument("--points", default="data/series_points.csv")
    ap.add_argument("--series", default="data/series_labeled.csv")
    ap.add_argument("--fits", default="data/fits_labeled.csv")
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()

    lab = pd.read_csv(a.labels)
    plan = pd.read_csv(a.plan)[["video_id", "bib", "pos", "official_forty",
                                "drafted"]]
    lab = lab.merge(plan, on=["video_id", "bib"], how="left")

    pts, fits, series = [], [], []
    for (vid, bib), g in lab.groupby(["video_id", "bib"]):
        g = g.sort_values("yard")
        row = g.iloc[0]
        pts.append(g[["video_id", "bib", "player_name", "cls", "split", "pos",
                      "drafted", "official_forty", "t_rel_s", "yard"]])
        if len(g) < MIN_MARKS:
            print(f"skip {row.player_name}: {len(g)} marks", file=sys.stderr)
            continue
        t = g.t_rel_s.to_numpy(float)
        x = g.yard.to_numpy(float)
        vmax, tau, t0, rmse, worst = fit(t, x)
        grid_fine = np.arange(0, 9, 1e-3)
        t40 = float(np.interp(40.0, model(np.maximum(grid_fine - t0, 0), vmax, tau),
                              grid_fine))
        fits.append({
            "video_id": vid, "bib": bib, "player_name": row.player_name,
            "cls": row.cls, "split": row.split, "pos": row.pos,
            "drafted": row.drafted, "official_forty": row.official_forty,
            "n_marks": len(g), "v_max_yd_s": vmax,
            "v_max_m_s": vmax * 0.9144, "tau_s": tau, "onset_s": t0,
            "rmse_yd": rmse, "max_resid_yd": worst,
            "t40_fit": t40, "t40_err": t40 - row.official_forty,
        })
        grid = np.arange(0.0, t.max() + 1e-9, a.dt)
        e = np.maximum(grid - t0, 0)
        xs = model(e, vmax, tau)
        v = vmax * (1 - np.exp(-e / tau))
        acc = (vmax / tau) * np.exp(-e / tau)
        series.append(pd.DataFrame({
            "video_id": vid, "bib": bib, "player_name": row.player_name,
            "cls": row.cls, "split": row.split, "t_s": grid, "x_yd": xs,
            "v_yd_s": v, "a_yd_s2": acc,
        }))

    for path, df in ((a.points, pd.concat(pts) if pts else pd.DataFrame()),
                     (a.fits, pd.DataFrame(fits)),
                     (a.series, pd.concat(series) if series else pd.DataFrame())):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    f = pd.DataFrame(fits)
    print(f"{len(pts)} runs labelled, {len(f)} fitted -> {a.fits}")
    if len(f):
        print(f.groupby(["cls", "split"]).agg(
            n=("bib", "size"), marks=("n_marks", "mean"),
            rmse=("rmse_yd", "median"), onset=("onset_s", "mean"),
            t40_err=("t40_err", lambda s: f"{s.mean():+.3f}")).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
