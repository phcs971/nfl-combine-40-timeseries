"""Recover a runner's position-time series from a tracking-camera run.

The camera pans to follow the athlete, so image position alone says nothing about
ground covered. The lane's black dashes are fixed in the world, so their frame-to-
frame shift measures the pan, and the runner's displacement is what is left over:

    displacement = (shift of the foot in image) - (shift of the marks in image)

projected onto the lane axis. Scale is taken from local mark spacing near the
runner, which keeps perspective from biasing the far end of the run.
"""

import argparse
import sys

import cv2
import numpy as np
from scipy.optimize import curve_fit

FIELD_H = 790          # rows above the broadcast panel
CENTRE = (0.30, 0.80)  # the tracking camera holds the runner near mid-frame


def lane_axis(hsv):
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    lane = ((s < 70) & (v > 110)).astype(np.uint8)
    lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    lane = cv2.morphologyEx(lane, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(lane)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == i).astype(np.uint8)
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(float)
    mu = pts.mean(0)
    ax = np.linalg.svd(pts - mu, full_matrices=False)[2][0]
    if ax[0] < 0:
        ax = -ax
    return m, mu, ax


def detect(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    la = lane_axis(hsv)
    if la is None:
        return None
    m, mu, ax = la
    band = cv2.dilate(m, np.ones((41, 41), np.uint8))
    h, s, v = (hsv[..., i].astype(int) for i in range(3))

    dark = ((v < 130) & (band > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    n, lab, st, ct = cv2.connectedComponentsWithStats(dark)
    perp_ax = np.array([-ax[1], ax[0]])
    marks, perps = [], []
    for j in range(1, n):
        a = st[j, cv2.CC_STAT_AREA]
        x, y, w, hh = st[j, :4]
        if a > 250 and 60 < w < 900 and hh < 45:
            marks.append(float((ct[j] - mu) @ ax))
            perps.append(float((ct[j] - mu) @ perp_ax))

    lo, hi = int(CENTRE[0] * frame.shape[1]), int(CENTRE[1] * frame.shape[1])
    shoe = ((h > 22) & (h < 45) & (s > 110) & (v > 150) & (band > 0)).astype(np.uint8)
    shoe[:, :lo] = 0
    shoe[:, hi:] = 0
    shoe = cv2.morphologyEx(shoe, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n2, lab2, st2, ct2 = cv2.connectedComponentsWithStats(shoe)
    blobs = [(ct2[j], st2[j, cv2.CC_STAT_AREA]) for j in range(1, n2)
             if st2[j, cv2.CC_STAT_AREA] > 60]
    if not blobs:
        return None
    # The grounded foot is the lowest one; it shares the ground plane with the
    # marks, so their coincidence carries no parallax term.
    c = max(blobs, key=lambda b: b[0][1])[0]
    return float((c - mu) @ ax), np.array(sorted(marks)), row_spacing(marks, perps)


def row_spacing(marks, perps) -> float:
    """Median gap within a single row of dashes.

    The lane carries two rows, offset from each other along its length. Pooling
    them and taking the median gap mixes true spacings with the offset between
    rows, and the mixture shifts as rows enter and leave frame - which distorts
    the recovered speed profile rather than merely rescaling it.
    """
    if len(marks) < 4:
        return float("nan")
    m = np.asarray(marks)
    p = np.asarray(perps)
    sp = []
    for sel in (p >= np.median(p), p < np.median(p)):
        r = np.sort(m[sel])
        if len(r) >= 3:
            sp.append(np.median(np.diff(r)))
    return float(np.median(sp)) if sp else float("nan")


def pan_shift(prev: np.ndarray, cur: np.ndarray, max_shift: float = 200.0):
    """Median nearest-neighbour displacement of world-fixed marks."""
    if len(prev) < 3 or len(cur) < 3:
        return np.nan
    d = []
    for p in prev:
        k = cur[np.argmin(np.abs(cur - p))]
        if abs(k - p) < max_shift:
            d.append(k - p)
    return float(np.median(d)) if len(d) >= 3 else np.nan


def run(video: str, t0: float, t1: float, fps: float | None = None):
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = 1 if fps is None else max(1, int(round(src_fps / fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)

    rows, prev, k = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > t1:
            break
        if k % step == 0:
            d = detect(frame[:FIELD_H])
            if d is not None:
                foot, marks, spacing = d
                shift = pan_shift(prev[1], marks) if prev else np.nan
                if prev and np.isfinite(shift):
                    rows.append((t, foot - prev[0] - shift, spacing))
                prev = (foot, marks)
        k += 1
    cap.release()
    return rows


def sprint_model(t, vmax, tau):
    """Mono-exponential sprint model (Samozino/Morin)."""
    return vmax * (t + tau * np.exp(-t / tau) - tau)


def fit_sprint(rel, yards):
    p, _ = curve_fit(sprint_model, rel, yards, p0=[9.0, 1.2], maxfev=20000)
    pred = sprint_model(rel, *p)
    rmse = float(np.sqrt(np.mean((yards - pred) ** 2)))
    return float(p[0]), float(p[1]), rmse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--total", type=float, default=40.0, help="run length, yards")
    args = ap.parse_args()

    rows = run(args.video, args.t0, args.t1)
    if not rows:
        print("no usable frames", file=sys.stderr)
        return 1
    t = np.array([r[0] for r in rows])
    dpx = np.array([r[1] for r in rows])
    sp = np.array([r[2] for r in rows])
    good = np.isfinite(dpx) & np.isfinite(sp) & (sp > 30)
    t, dpx, sp = t[good], dpx[good], sp[good]
    # Repair, do not drop, frames whose displacement is not physically reachable.
    # Dropping one deletes its displacement from the cumulative sum and shortens
    # the run; interpolating keeps the time base and the distance intact.
    step = dpx / sp
    med = np.median(step[step > 0])
    bad = ~((step > -0.2 * med) & (step < 3.0 * med))
    if bad.any() and (~bad).sum() > 4:
        step[bad] = np.interp(t[bad], t[~bad], step[~bad])
    n_bad = int(bad.sum())

    # Distance in mark-spacings, so perspective cancels; the yard value of one
    # spacing follows from the run being 40 yards.
    pos_units = np.cumsum(step)
    yards = pos_units / pos_units[-1] * args.total
    print(f"frames used: {len(t)} ({n_bad} repaired)   span {t[0]:.2f}-{t[-1]:.2f}s")
    print(f"one mark spacing = {args.total / pos_units[-1]:.3f} yd")
    rel = t - args.t0
    for frac in (0.25, 0.5, 0.75, 1.0):
        i = int(frac * (len(t) - 1))
        print(f"  t={rel[i]:5.2f}s  {yards[i]:6.2f} yd")
    for y in (10.0, 20.0):
        if yards[-1] >= y:
            print(f"  reaches {y:.0f} yd at t={np.interp(y, yards, rel):.3f}s")
    vmax, tau, rmse = fit_sprint(rel, yards)
    print(f"\nsprint fit: v_max {vmax * 0.9144:.2f} m/s   tau {tau:.3f} s   "
          f"rmse {rmse:.3f} yd")
    for y in (10.0, 40.0):
        lo, hi = 0.0, 12.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if sprint_model(mid, vmax, tau) < y:
                lo = mid
            else:
                hi = mid
        print(f"  model reaches {y:.0f} yd at t={(lo + hi) / 2:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
