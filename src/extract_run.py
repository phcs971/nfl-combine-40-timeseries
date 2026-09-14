"""Extract the frames of each planned run, plus the metadata the labeller needs."""

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

FPS = 30000 / 1001
MARKS = [5, 10, 15, 20, 25, 30, 35, 40]
PRE, POST = 0.6, 1.6


def model(t, vmax, tau):
    return vmax * (t + tau * np.exp(-t / tau) - tau)


def seed_times(final_clock, vmax, tau):
    """Predicted crossing time per mark, used to park the viewer near the frame."""
    if not np.isfinite(vmax) or not np.isfinite(tau):
        tau = 0.9
        vmax = float(least_squares(
            lambda v: model(final_clock, v[0], tau) - 40.0, [9.0],
            bounds=([5], [14])).x[0])
    out = {}
    for y in MARKS:
        try:
            out[y] = round(brentq(lambda t: model(t, vmax, tau) - y, 1e-4, 12.0), 4)
        except ValueError:
            pass
    return out


def extract(video: pathlib.Path, row, root: pathlib.Path, quality: int) -> dict:
    start = float(row.t_zero) - PRE
    dur = float(row.final_clock) + PRE + POST
    frames = root / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("*.jpg"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
         "-i", str(video), "-t", f"{dur:.3f}", "-fps_mode", "passthrough",
         "-q:v", str(quality), str(frames / "f%04d.jpg")], check=True)
    n = len(list(frames.glob("*.jpg")))
    meta = {
        "video_id": row.video_id, "athlete": row.player_name, "bib": int(row.bib),
        "cls": row.cls, "pos": row.pos, "split": row.split,
        "official_forty": float(row.official_forty),
        "drafted": bool(row.drafted), "fps": FPS, "clip_start": round(start, 4),
        "t_zero": float(row.t_zero), "final_clock": float(row.final_clock),
        "n": n, "marks": MARKS,
        "seed": seed_times(float(row.final_clock), row.v_max_yd_s, row.tau_s),
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/label_plan.csv")
    ap.add_argument("--video-dir", default="video")
    ap.add_argument("--out", default="label")
    ap.add_argument("--quality", type=int, default=3)
    ap.add_argument("--only", help="video_id or 'video_id:bib' to limit to")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    plan = pd.read_csv(a.plan)
    if a.only:
        vid, _, bib = a.only.partition(":")
        plan = plan[plan.video_id == vid]
        if bib:
            plan = plan[plan.bib == int(bib)]

    out = pathlib.Path(a.out)
    done = skipped = 0
    for row in plan.itertuples():
        root = out / f"{row.video_id}_{int(row.bib):02d}"
        video = pathlib.Path(a.video_dir) / f"{row.video_id}.mp4"
        if not video.exists():
            print(f"missing video {video}", file=sys.stderr)
            continue
        if (root / "meta.json").exists() and not a.force:
            skipped += 1
            continue
        m = extract(video, row, root, a.quality)
        done += 1
        print(f"{root.name:22s} {m['athlete']:24s} {m['n']:4d} frames")
    print(f"\n{done} extracted, {skipped} already present -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
