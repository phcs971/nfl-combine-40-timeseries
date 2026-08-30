"""Fit a sprint model per run using mat crossings plus the panel's finish time.

The numbered mats are 10 yards apart but their first mat is not the dash's 10-yard
mark, so their distance from the start is carried as one unknown per video. It is
the same physical offset for every run in that video, which makes agreement between
independently fitted runs a check on the geometry.
"""

import argparse
import sys

import numpy as np
from itertools import combinations
import pandas as pd
from scipy.optimize import least_squares

from crossings import gather, crossings


def model(t, vmax, tau):
    return vmax * (t + tau * np.exp(-t / tau) - tau)


def fit_run(cross_rel, t_final, offset=None, k=None, k0=0):
    """Fit v_max, tau (and the mat offset when not supplied).

    `k` gives each crossing's mat index; `k0` is the shorthand for consecutive
    mats starting there. A run that misses the first mat otherwise has every
    crossing assigned ten yards short.
    """
    k = np.arange(len(cross_rel)) + k0 if k is None else np.asarray(k)

    def resid(p):
        vmax, tau = p[0], p[1]
        off = p[2] if offset is None else offset
        r = [model(t_final, vmax, tau) - 40.0]
        r += list(model(cross_rel, vmax, tau) - (off + 10.0 * k))
        return r

    p0 = [9.0, 0.8] + ([15.0] if offset is None else [])
    lo = [5, 0.2] + ([0] if offset is None else [])
    hi = [13, 3] + ([35] if offset is None else [])
    s = least_squares(resid, p0, bounds=(lo, hi))
    vmax, tau = s.x[0], s.x[1]
    off = s.x[2] if offset is None else offset
    return vmax, tau, off, float(np.max(np.abs(s.fun)))


def best_assignment(cross_rel, t_final, offset, n_mats: int = 4,
                    allow_drop: bool = True):
    """Fit over which mats the crossings correspond to, not just where they start.

    Crossings are not necessarily consecutive mats: one can be missed when the
    runner occludes it, and a stray detection can add one that is not a mat at
    all. Forcing them to be consecutive leaves those runs unfittable, which is
    what put the failures in a tight band near half a mat spacing rather than
    scattering them.
    """
    best = None
    idx = range(len(cross_rel))
    subsets = [tuple(idx)]
    if allow_drop and len(cross_rel) >= 3:
        subsets += [tuple(j for j in idx if j != d) for d in idx]

    for keep in subsets:
        rel = np.asarray([cross_rel[j] for j in keep])
        if len(rel) < 2:
            continue
        for combo in combinations(range(n_mats), len(rel)):
            vmax, tau, off, res = fit_run(rel, t_final, offset=offset, k=combo)
            # Prefer explanations that discard nothing, all else equal.
            penalty = 0.15 * (len(cross_rel) - len(rel))
            if best is None or res + penalty < best[4]:
                best = (vmax, tau, off, res, res + penalty, keep, combo)
    if best is None:
        return None
    vmax, tau, off, res, _, keep, combo = best
    return vmax, tau, off, res


def time_at(y, vmax, tau):
    lo, hi = 0.0, 15.0
    for _ in range(80):
        m = (lo + hi) / 2
        if model(m, vmax, tau) < y:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/runs.csv")
    ap.add_argument("--video-dir", default="video")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.runs, dtype={"video_id": str})
    df = df[(df.video_id == args.video_id) & df.complete & ~df.duplicate]
    df = df.sort_values("t_start").head(args.limit)
    path = f"{args.video_dir}/{args.video_id}.mp4"

    # Pass 1: collect crossings, and fit the video's mat offset from the runs
    # that saw enough mats to constrain it.
    obs = []
    for _, r in df.iterrows():
        t0 = r.t_zero
        ev = crossings(gather(path, t0, t0 + r.final_clock + 1.5))
        rel = np.array([e - t0 for e in ev])
        rel = rel[(rel > 0.3) & (rel < r.final_clock + 0.5)]
        obs.append((r.player_name, rel, r.final_clock))

    solid = [fit_run(rel, fc) for _, rel, fc in obs if len(rel) >= 4]
    solid = [s for s in solid if s[3] < 1.0]
    if not solid:
        print("no run constrains the mat offset", file=sys.stderr)
        return 1
    offset = float(np.median([s[2] for s in solid]))
    print(f"mat offset fitted from {len(solid)} run(s): {offset:.2f} yd "
          f"(sd {np.std([s[2] for s in solid]):.3f})\n")

    # Pass 2: the offset is one physical constant for the video, so holding it
    # fixed lets runs that saw only two or three mats still be fitted.
    print(f"{'athlete':22} {'n_x':>3} {'v_max':>6} {'tau':>5} {'resid':>6} "
          f"{'40yd':>5}")
    keep = []
    for name, rel, fc in obs:
        if len(rel) < 2:
            print(f"{str(name)[:22]:22} {len(rel):3d}   (too few crossings)")
            continue
        got = best_assignment(rel, fc, offset)
        if got is None:
            print(f"{str(name)[:22]:22} {len(rel):3d}   (no assignment)")
            continue
        vmax, tau, _, res = got
        keep.append((name, vmax, tau, res))
        print(f"{str(name)[:22]:22} {len(rel):3d} {vmax * 0.9144:6.2f} {tau:5.2f} "
              f"{res:6.3f} {fc:5.2f}")
    if keep:
        v = np.array([k[1] for k in keep]) * 0.9144
        print(f"\nv_max across runs: {v.mean():.2f} +/- {v.std():.2f} m/s "
              f"(range {v.min():.2f}-{v.max():.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
