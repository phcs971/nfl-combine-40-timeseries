"""Animate the yardage detector over one run, to watch tracking hold or slip.

The check the animation makes visible: a tick is fixed in the world, so its index
must stay with the same physical mark while the camera pans across it. Numbers
that flicker or renumber as marks cross the frame are the tracker slipping.
"""

import argparse
import pathlib
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

from yardage import (lane, detect_ticks, detect_mats, detect_yard_lines,
                     YardTracker, FIELD_H)

CYAN, TICK, GRN, LINE = (60, 230, 255), (60, 120, 255), (80, 255, 140), (255, 90, 220)


def plate(im, text, org, color, scale=0.8, thick=2):
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    x, y = org
    cv2.rectangle(im, (x - 8, y - h - 8), (x + w + 8, y + b + 4), (12, 13, 16), -1)
    cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick,
                cv2.LINE_AA)


def render(video, t0, t1, outdir, width=720, stride=3):
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    tr = YardTracker()
    k = n = 0
    mat_log = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > t1:
            break
        k += 1
        if k % stride:
            continue
        f = fr[:FIELD_H]
        L = lane(f)
        if L is None:
            continue
        mask, mu, ax = L
        ticks = detect_ticks(f, ax, mask)
        tracked = tr.update([z[1] for z in ticks]) if len(ticks) >= 3 else {}
        mats = detect_mats(f, ax, mask)
        lines = detect_yard_lines(f, ax)

        v = f.copy()
        for ctr, d, _ in lines:
            a, b = (ctr - d * 1400).astype(int), (ctr + d * 1400).astype(int)
            cv2.line(v, tuple(a), tuple(b), LINE, 3, cv2.LINE_AA)
        pos2idx = {round(p, 1): i for i, p in tracked.items()}
        for box, along, _ in ticks:
            cv2.polylines(v, [box.astype(int)], True, TICK, 3, cv2.LINE_AA)
            i = pos2idx.get(round(along, 1))
            if i is not None:
                plate(v, str(i), (int(box[:, 0].mean()) - 12,
                                  int(box[:, 1].max()) + 34), (150, 200, 255), 0.7, 2)
        for b in mats:
            cv2.polylines(v, [b.astype(int)], True, CYAN, 5, cv2.LINE_AA)
            cor = sorted(b, key=lambda q: float(q @ ax))[:2]
            cv2.line(v, tuple(np.int32(cor[0])), tuple(np.int32(cor[1])), GRN, 8,
                     cv2.LINE_AA)
            mi = tr.index_of(float(np.min(np.asarray(b) @ ax)))
            if np.isfinite(mi):
                mat_log.append(mi)
                plate(v, f"mat @ {mi:.1f}", (int(cor[0][0]) - 40,
                                             int(min(c[1] for c in cor)) - 18),
                      GRN, 0.85, 2)
        plate(v, f"t = {t - t0:5.2f}s", (24, 52), (240, 240, 245), 0.95, 2)
        if tracked:
            plate(v, f"ticks {len(ticks):2d}   idx {min(tracked)}..{max(tracked)}",
                  (24, 108), (150, 200, 255), 0.8, 2)
        h = int(v.shape[0] * width / v.shape[1])
        cv2.imwrite(str(out / f"f{n:04d}.png"), cv2.resize(v, (width, h)))
        n += 1
    cap.release()
    return n, mat_log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", default="Sr-Q6UjJq6g")
    ap.add_argument("--athlete", default="Cameron Ball")
    ap.add_argument("--out", default="frames/validation.gif")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    fits = pd.read_csv("data/fits.csv", dtype={"video_id": str})
    r = fits[(fits.video_id == args.video_id) & (fits.player_name == args.athlete)
             & (fits.quality == "ok")].sort_values("t_zero").iloc[0]
    t0 = float(r.t_zero)
    tmp = "/private/tmp/claude-502/-Users-pedro-soares-Documents-Code-nfl-combine-40-timeseries/bce26fa5-1ff4-4692-860d-3e4074d04dbd/scratchpad/gifsrc"
    n, mats = render(f"video/{args.video_id}.mp4", t0, t0 + r.final_clock + 1.2, tmp)
    print(f"{n} frames rendered", file=sys.stderr)

    pal = f"{tmp}/pal.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{tmp}/f%04d.png",
                    "-vf", "palettegen=max_colors=64", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", f"{tmp}/f%04d.png", "-i", pal, "-lavfi",
                    "paletteuse=dither=bayer:bayer_scale=3", args.out], check=True)
    size = pathlib.Path(args.out).stat().st_size / 1e6
    print(f"{args.out}  {size:.2f} MB  ({n} frames @ {args.fps}fps)")
    if mats:
        m = np.array(mats)
        print(f"mat index observed {len(m)}x: {m.min():.1f}..{m.max():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
