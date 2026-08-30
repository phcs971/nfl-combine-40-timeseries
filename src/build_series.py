"""Fit every run and emit position-time series.

Per video the mat offset is fitted once from the runs that saw enough mats to
constrain it, then held fixed so sparser runs can still be fitted.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from crossings import gather, crossings
from fit_runs import fit_run, best_assignment, model, time_at

GOOD_RESID = 0.8


def process_video(vid: str, runs: pd.DataFrame, path: str):
    obs = []
    for _, r in runs.iterrows():
        t0 = r.t_zero
        try:
            ev = crossings(gather(path, t0, t0 + r.final_clock + 1.5))
        except Exception as exc:                      # noqa: BLE001
            print(f"    {r.player_name}: {exc}", file=sys.stderr)
            ev = []
        rel = np.array([e - t0 for e in ev])
        rel = rel[(rel > 0.3) & (rel < r.final_clock + 0.5)]
        obs.append((r, rel))

    solid = [fit_run(rel, r.final_clock) for r, rel in obs if len(rel) >= 4]
    solid = [s for s in solid if s[3] < 1.0]
    if not solid:
        return None, []
    offset = float(np.median([s[2] for s in solid]))

    out = []
    for r, rel in obs:
        if len(rel) < 2:
            out.append((r, rel, None))
            continue
        out.append((r, rel, best_assignment(rel, r.final_clock, offset)))
    return offset, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/runs.csv")
    ap.add_argument("--video-dir", default="video")
    ap.add_argument("--fits", default="data/fits.csv")
    ap.add_argument("--series", default="data/series.csv")
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.runs, dtype={"video_id": str})
    df = df[df.complete & ~df.duplicate]

    fits, series = [], []
    for vid, g in df.groupby("video_id"):
        path = f"{args.video_dir}/{vid}.mp4"
        if not pathlib.Path(path).exists():
            continue
        print(f"{vid}: {len(g)} runs", file=sys.stderr)
        offset, out = process_video(vid, g, path)
        if offset is None:
            print("    no run constrains the mat offset", file=sys.stderr)
            continue
        ok = 0
        for r, rel, fit in out:
            row = {
                "video_id": vid, "cls": r.cls, "player_name": r.player_name,
                "bib": r.bib, "t_zero": r.t_zero, "final_clock": r.final_clock,
                "official_forty": r.official_forty,
                "n_crossings": len(rel), "mat_offset": round(offset, 3),
            }
            if fit is not None:
                vmax, tau, _, res = fit
                row |= {"v_max_yd_s": round(vmax, 4),
                        "v_max_m_s": round(vmax * 0.9144, 3),
                        "tau_s": round(tau, 4), "resid_yd": round(res, 4),
                        "t_10yd": round(time_at(10, vmax, tau), 4),
                        "quality": "ok" if res < GOOD_RESID else "poor"}
                if res < GOOD_RESID:
                    ok += 1
                    for t_ in np.arange(0.0, r.final_clock + 1e-9, args.dt):
                        series.append({
                            "video_id": vid, "player_name": r.player_name,
                            "t_s": round(float(t_), 3),
                            "x_yd": round(float(model(t_, vmax, tau)), 4),
                        })
            else:
                row |= {"quality": "unfitted"}
            fits.append(row)
        print(f"    offset {offset:.2f} yd, {ok}/{len(out)} good", file=sys.stderr)

    pd.DataFrame(fits).to_csv(args.fits, index=False)
    pd.DataFrame(series).to_csv(args.series, index=False)
    print(f"\n{len(fits)} runs -> {args.fits}")
    print(f"{len(series)} samples -> {args.series}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
