"""Times at which a runner's foot passes each numbered distance mat.

A crossing is a coincidence of runner and mat inside one frame, so it is immune to
the camera panning, and both lie on the ground plane so it carries no parallax
term. Mats are identified by the order they are passed rather than by reading
their numerals.
"""

import argparse
import sys

import cv2
import numpy as np

from detectors import detect_athlete, detect_mats, lane_axis_of, FIELD_H


def gather(video: str, t0: float, t1: float):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > t1:
            break
        f = frame[:FIELD_H]
        la = lane_axis_of(f)
        at = detect_athlete(f)
        if la is None or at is None:
            continue
        mu, d = la
        foot = at[0]
        mats = merge_mats([(float((c - mu) @ d), w) for c, w in detect_mats(f)])
        rows.append((t, float((foot - mu) @ d), mats))
    cap.release()
    return rows


def merge_mats(mats, tol: float = 300.0):
    """Combine blobs belonging to one mat.

    A mat's white numerals split its dark face, so it is detected as two blobs a
    couple of hundred pixels apart. Ten yards is an order of magnitude wider than
    that, so anything closer is the same mat; left separate, each mat yields two
    crossings and the run's mats can no longer be counted.
    """
    if not mats:
        return []
    mats = sorted(mats)
    out, cur = [], [mats[0]]
    for m in mats[1:]:
        if m[0] - cur[-1][0] < tol:
            cur.append(m)
        else:
            out.append((float(np.mean([c[0] for c in cur])),
                        int(sum(c[1] for c in cur))))
            cur = [m]
    out.append((float(np.mean([c[0] for c in cur])),
                int(sum(c[1] for c in cur))))
    return out


def crossings(rows, gap: float = 0.25):
    """Interpolate the time the foot passes each mat, mats taken in order."""
    events = []
    for (ta, fa, ma), (tb, fb, mb) in zip(rows, rows[1:]):
        if tb - ta > gap or not ma or not mb:
            continue
        for sa, wa in ma:
            # Match the same mat in the next frame by nearest projected position.
            if not mb:
                continue
            sb, _ = min(mb, key=lambda m: abs(m[0] - sa))
            if abs(sb - sa) > 160:
                continue
            da, db = fa - sa, fb - sb
            if da < 0 <= db:
                frac = -da / (db - da) if db != da else 0.0
                events.append(ta + frac * (tb - ta))
    events.sort()
    merged = []
    for e in events:
        if not merged or e - merged[-1] > 0.30:
            merged.append(e)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--split10", type=float, default=None)
    ap.add_argument("--forty", type=float, default=None)
    args = ap.parse_args()

    rows = gather(args.video, args.t0, args.t1)
    n_mat = sum(1 for r in rows if r[2])
    print(f"frames {len(rows)} usable, {n_mat} with a mat visible")
    ev = crossings(rows)
    print(f"crossings: {[round(e - args.t0, 3) for e in ev]}")
    for i, e in enumerate(ev):
        yd = 10 * (i + 1)
        line = f"  {yd:2d} yd at t={e - args.t0:.3f}s"
        if yd == 10 and args.split10:
            line += f"   panel {args.split10:.2f}  d={e - args.t0 - args.split10:+.3f}"
        if args.forty and yd == 40:
            line += f"   panel {args.forty:.2f}  d={e - args.t0 - args.forty:+.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
