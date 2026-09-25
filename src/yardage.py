"""Runner position along the lane on every frame, in lane periods then yards.

World frame on the ground plane: s runs along the lane in dash periods, w across it
(near row w=0, far row w=1). The runner is located against dashes visible in the
same frame, so nothing accumulates across frames.

Row indices from `lane.detect` are per-frame arbitrary; they are made world-consistent
by requiring the camera to move less than half a period between frames. s=0 is the
start line on the far row; the cross-lane direction comes from the painted yard lines.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from lane import LaneFrame, Row, intersect

ROOT = Path(__file__).resolve().parents[1]
FPS = 30000 / 1001
L_SHO, R_SHO, L_HIP, R_HIP, L_ANK, R_ANK, L_WRI, R_WRI = 5, 6, 11, 12, 15, 16, 9, 10
KP_MIN = 0.3
ROW_RESID_MAX = 0.08
SPIKE = 0.2  # periods


def load(run_id: str) -> list[dict]:
    # Local cache written by extract.py, never fetched from elsewhere.
    with open(ROOT / "cache/frames" / f"{run_id}.pkl", "rb") as f:
        return pickle.load(f)


def _row_at_x(row: Row, x: float) -> np.ndarray:
    t = (x - row.p0[0]) / row.d[0]
    return row.p0 + t * row.d


def _rows(lf: LaneFrame) -> list[Row] | None:
    if len(lf.rows) != 2 or any(r.resid > ROW_RESID_MAX for r in lf.rows):
        return None
    return lf.rows


def unwrap(fr: list[dict]) -> np.ndarray:
    """World offset per frame and row, shape (n, 2); nan where the lane is unusable."""
    n = len(fr)
    off = np.full((n, 2), np.nan)
    for r in range(2):
        prev_s, prev_k, vel = None, None, 0.0
        for k in range(n):
            rows = _rows(fr[k]["lane"])
            if rows is None:
                continue
            row = rows[r]
            s_raw = float(row.s_of_u(row.u(_row_at_x(row, 960.0))[0]))
            if prev_s is None:
                o = 0.0
            else:
                g = k - prev_k
                if g > 8:
                    # Too long a gap to trust the half-period rule.
                    break
                o = np.round(prev_s + vel * g - s_raw)
            s_w = s_raw + o
            if prev_s is not None:
                vel = 0.5 * vel + 0.5 * (s_w - prev_s) / (k - prev_k)
            prev_s, prev_k = s_w, k
            off[k, r] = o
    return off


@dataclass
class Rails:
    """Lane coordinates for one frame, anchored on the far dash row.

    The runner runs along the far row, so that row's exact 1-D projective map carries
    the measurement. A rung is the image of a line across the lane: its direction comes
    from the painted yard lines and it ends on the near row's line.
    """
    near: Row
    far: Row
    of: float     # world s = far index + of
    theta: float  # rung angle from the lane direction, deg, upward

    def pf(self, s):
        return self.far.point(np.atleast_1d(np.asarray(s, float)) - self.of)

    def rung(self, s) -> np.ndarray:
        """Vector from the near row's line to the far row at s, per s."""
        t = self.far.d
        th = np.radians(self.theta)
        d = np.cos(th) * t + np.sin(th) * np.array([t[1], -t[0]])
        P = self.pf(s)
        M = np.array([d, -self.near.d]).T
        tt = (np.linalg.inv(M) @ (self.near.p0 - P).T)[0]
        return -tt[:, None] * d

    def point(self, s, w) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, float))
        w = np.asarray(w, float)
        w = w[..., None] if w.ndim else w
        return self.pf(s) + (w - 1) * self.rung(s)

    def _grid(self):
        lo = self.far.idx.min() + self.of - 3
        hi = self.far.idx.max() + self.of + 3
        return np.arange(lo, hi, 0.02)

    @staticmethod
    def _root(g, f):
        sc = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) <= 0)[0]
        if not len(sc):
            return None
        i = sc[0]
        return g[i] + (g[i + 1] - g[i]) * f[i] / (f[i] - f[i + 1] + 1e-12)

    def s_through(self, p) -> tuple[float, float]:
        """(s, w) of the rung line passing through image point p."""
        g = self._grid()
        a, d = self.pf(g), self.rung(g)
        f = d[:, 0] * (p[1] - a[:, 1]) - d[:, 1] * (p[0] - a[:, 0])
        s = self._root(g, f)
        if s is None:
            return np.nan, np.nan
        a, d = self.pf(s)[0], self.rung(s)[0]
        return float(s), float(1 + (p - a) @ d / (d @ d))

    def s_below(self, p, w: float) -> float:
        """s of the ground point at lateral position w straight below p in the image."""
        g = self._grid()
        s = self._root(g, self.point(g, w)[:, 0] - p[0])
        return np.nan if s is None else float(s)

    def rung_angle(self, s: float) -> float:
        """Angle (deg) from the lane direction to the rung, near -> far, positive upward in the image."""
        d = self.rung(s)[0]
        t = self.far.d
        return float(-np.degrees(np.arctan2(t[0] * d[1] - t[1] * d[0], t @ d)))


def rails(fr_k: dict, off_k: np.ndarray, theta: float) -> Rails | None:
    rows = _rows(fr_k["lane"])
    if rows is None or np.isnan(off_k[1]) or np.isnan(theta):
        return None
    return Rails(rows[0], rows[1], float(off_k[1]), float(theta))


def yard_line_hits(lf: LaneFrame, far: Row) -> np.ndarray:
    """Painted yard lines where they meet the far row: (u along the row, angle), one per line."""
    t = far.d
    hits = []
    for x0, y0, x1, y1 in lf.yard_lines:
        a, b = np.array([x0, y0], float), np.array([x1, y1], float)
        d = (b - a) / np.linalg.norm(b - a)
        p = intersect(a, d, far.p0, far.d)
        # A yard line runs down to the lane; other white edges (numbers, logos) rarely do.
        if p is None or min(np.linalg.norm(a - p), np.linalg.norm(b - p)) > 250:
            continue
        v = d if d[1] < 0 else -d
        hits.append((float(far.u(p)[0]), float(-np.degrees(np.arctan2(t[0] * v[1] - t[1] * v[0], t @ v)))))
    if not hits:
        return np.zeros((0, 2))
    h = np.array(sorted(hits))
    # Hough returns several segments per painted line.
    groups = np.split(h, np.nonzero(np.diff(h[:, 0]) > 25)[0] + 1)
    c = np.array([np.median(g, 0) for g in groups])
    return c[np.abs(c[:, 1] - np.median(c[:, 1])) < 20]


def yard_theta(lf: LaneFrame, far: Row, u_ref: float) -> float:
    """Rung angle at far-row position u_ref from the painted yard lines, or nan."""
    c = yard_line_hits(lf, far)
    if not len(c):
        return np.nan
    if len(c) == 1:
        return float(c[0, 1])
    # The angle varies smoothly along the row (the lines meet at a vanishing point).
    slope, icpt = np.polyfit(c[:, 0], c[:, 1], 1)
    u = np.clip(u_ref, c[:, 0].min() - 300, c[:, 0].max() + 300)
    return float(slope * u + icpt)


def _hands_s(fr: list[dict], off: np.ndarray, trk: "Track", k0: float) -> float:
    """Near-row s straight below the runner's hands just before the clock starts."""
    vals = []
    for k in range(max(0, int(k0) - 10), min(len(fr), int(k0) + 1)):
        rows = _rows(fr[k]["lane"])
        if rows is None or np.isnan(off[k, 0]):
            continue
        for j in (L_WRI, R_WRI):
            if trk.kpts[k, j, 2] > KP_MIN:
                q = _row_at_x(rows[0], trk.kpts[k, j, 0])
                vals.append(float(rows[0].s_of_u(rows[0].u(q)[0])) + off[k, 0])
    return float(np.median(vals)) if vals else np.nan


def start_line(fr: list[dict], off: np.ndarray, k0: float, trk: "Track") -> tuple[float, int]:
    """(start line s on the far row, support frames)."""
    cands = []
    lo, hi = max(0, int(k0) - 40), min(len(fr), int(k0) + 45)
    for k in range(lo, hi):
        rows = _rows(fr[k]["lane"])
        if rows is None or np.isnan(off[k]).any():
            continue
        near, far = rows
        for b in fr[k]["lane"].transverse:
            d = b.ends[1] - b.ends[0]
            L = np.linalg.norm(d)
            d = d / L
            pn = intersect(b.ends[0], d, near.p0, near.d)
            pf = intersect(b.ends[0], d, far.p0, far.d)
            if pn is None or pf is None:
                continue
            tn, tf = (pn - b.ends[0]) @ d, (pf - b.ends[0]) @ d
            # The start line spans the lane, so it must reach across both rows.
            if not (-25 <= min(tn, tf) and max(tn, tf) <= L + 25):
                continue
            sn = float(near.s_of_u(near.u(pn)[0])) + off[k, 0]
            sf = float(far.s_of_u(far.u(pf)[0])) + off[k, 1]
            cands.append((k, sn, sf))
    if not cands:
        return np.nan, 0
    c = np.array(cands)
    # The start line is where the hands are set; other cross-lane marks (the 10-yd
    # timing gate) can be seen in more frames when the stance itself is hidden.
    hands = _hands_s(fr, off, trk, k0)
    if not np.isnan(hands):
        c = c[np.abs(c[:, 1] - hands) < 1.5]
    best, best_n = None, 0
    for s in c[:, 1]:
        m = np.abs(c[:, 1] - s) < 0.1
        n = len(np.unique(c[m, 0]))
        if n > best_n:
            best, best_n = m, n
    if best is None:
        return np.nan, 0
    return float(np.median(c[best, 2])), best_n


@dataclass
class Track:
    kpts: np.ndarray    # (n, 17, 3), nan where missing
    box: np.ndarray     # (n, 4)


def _between_rows(rows: list[Row], p: np.ndarray, margin: float = 0.6) -> bool:
    near, far = rows
    yn, yf = _row_at_x(near, p[0])[1], _row_at_x(far, p[0])[1]
    sep = yn - yf
    return yf - margin * sep <= p[1] <= yn + margin * sep


def _core(K: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, bool]:
    """Hip centre and shoulder centre (x, y, x, y); from the box when the keypoints are weak."""
    if min(K[L_HIP, 2], K[R_HIP, 2], K[L_SHO, 2], K[R_SHO, 2]) >= KP_MIN:
        return np.concatenate([(K[L_HIP, :2] + K[R_HIP, :2]) / 2, (K[L_SHO, :2] + K[R_SHO, :2]) / 2]), True
    cx, h = (box[0] + box[2]) / 2, box[3] - box[1]
    return np.array([cx, box[1] + 0.55 * h, cx, box[1] + 0.25 * h]), False


def track_runner(fr: list[dict], k0: float) -> Track:
    n = len(fr)
    kp = np.full((n, 17, 3), np.nan)
    bx = np.full((n, 4), np.nan)
    # Seed at the clock start: the runner is the person set on the lane.
    seed = None
    for k in sorted(range(max(0, int(k0) - 8), min(n, int(k0) + 3)), key=lambda k: abs(k - k0)):
        rows = _rows(fr[k]["lane"])
        if rows is None or not len(fr[k]["boxes"]):
            continue
        best = None
        for i, K in enumerate(fr[k]["kpts"]):
            pts = [K[j, :2] for j in (L_ANK, R_ANK, L_WRI, R_WRI) if K[j, 2] > KP_MIN]
            inside = sum(_between_rows(rows, p) for p in pts)
            h = fr[k]["boxes"][i, 3] - fr[k]["boxes"][i, 1]
            if inside >= 2 and (best is None or (inside, h) > best[0]):
                best = ((inside, h), i)
        if best:
            seed = (k, best[1])
            break
    if seed is None:
        return Track(kp, bx)
    k_s, i_s = seed
    kp[k_s], bx[k_s] = fr[k_s]["kpts"][i_s], fr[k_s]["boxes"][i_s]
    for step in (1, -1):
        last_k, last_c, vel = k_s, _core(kp[k_s], bx[k_s])[0], np.zeros(4)
        k = k_s + step
        while 0 <= k < n and abs(k - last_k) <= 30:
            Ks = fr[k]["kpts"]
            if len(Ks):
                g = abs(k - last_k)
                pred = last_c + vel * min(g, 3)
                # The torso is foreshortened in the stance; the box keeps the scale sensible.
                lb = bx[last_k]
                torso = max(np.hypot(*(pred[:2] - pred[2:])), 0.4 * np.sqrt((lb[2] - lb[0]) * (lb[3] - lb[1])))
                cores = [_core(K, B) for K, B in zip(Ks, fr[k]["boxes"])]
                # Hip and shoulder centres move smoothly and survive truncated boxes;
                # limbs and box extents swing with every stride.
                cost = np.array([(np.linalg.norm(c[:2] - pred[:2]) + np.linalg.norm(c[2:] - pred[2:])) / torso
                                 + (0.0 if good else 0.2) for c, good in cores])
                rows = _rows(fr[k]["lane"])
                if rows is not None:
                    for i, K in enumerate(Ks):
                        legs = [K[j, :2] for j in (13, 14, L_ANK, R_ANK) if K[j, 2] > KP_MIN]
                        if legs and not any(_between_rows(rows, q, margin=1.0) for q in legs):
                            cost[i] += 0.5
                i = int(np.argmin(cost))
                if cost[i] < (0.8 + 0.1 * g if g <= 3 else 1.3):
                    c = cores[i][0]
                    if g <= 3:
                        vel = 0.6 * vel + 0.4 * (c - last_c) / g
                    kp[k], bx[k] = Ks[i], fr[k]["boxes"][i]
                    last_k, last_c = k, c
            k += step
    return Track(kp, bx)


def _hip(K: np.ndarray) -> np.ndarray | None:
    if np.isnan(K).any() or min(K[L_HIP, 2], K[R_HIP, 2]) < KP_MIN:
        return None
    return (K[L_HIP, :2] + K[R_HIP, :2]) / 2


def _far_below(far: Row, of: float, p: np.ndarray) -> float:
    """World s on the far row straight below image point p."""
    return float(far.s_of_u(far.u(_row_at_x(far, p[0]))[0]) + of)


def _walk(valid: list[int], start: int):
    for step in (1, -1):
        yield [k for k in sorted(valid, key=lambda k: abs(k - start)) if (k - start) * step >= 0]


def fix_far(fr, off, trk, k_ref) -> np.ndarray:
    """Remove far-row slips: the runner cannot move a whole period in one frame."""
    valid = [k for k in range(len(fr)) if _rows(fr[k]["lane"]) is not None and not np.isnan(off[k, 1])]
    if not valid:
        return off
    start = min(valid, key=lambda k: abs(k - k_ref))
    fixed = off.copy()
    for ks in _walk(valid, start):
        corr, hist = 0.0, []
        for k in ks:
            far = _rows(fr[k]["lane"])[1]
            hip = _hip(trk.kpts[k])
            if hip is not None and len(hist) >= 2:
                hk, hs = np.array(hist[-6:]).T
                slope, icpt = np.polyfit(hk, hs, 1)
                corr += np.round(slope * k + icpt - _far_below(far, off[k, 1] + corr, hip))
            fixed[k, 1] = off[k, 1] + corr
            if hip is not None:
                hist.append((k, _far_below(far, fixed[k, 1], hip)))
    return fixed


def _robust_smooth(x: np.ndarray, med: int = 9, win: int = 15, hold: int = 15) -> np.ndarray:
    ok = ~np.isnan(x)
    if ok.sum() < 3:
        return x
    idx = np.arange(len(x))
    y = np.interp(idx, idx[ok], x[ok])
    y = np.array([np.median(y[max(0, i - med // 2):i + med // 2 + 1]) for i in range(len(y))])
    if len(y) >= win:
        y = savgol_filter(y, win, 2)
    # The camera pans smoothly: hold the edge values a little way past the last estimate.
    y[: max(0, idx[ok].min() - hold)] = np.nan
    y[idx[ok].max() + hold + 1:] = np.nan
    return y


def yard_rungs(fr: list[dict], off: np.ndarray, trk: "Track") -> tuple[np.ndarray, np.ndarray]:
    """Per-frame rung angle at the runner from the yard lines, raw and smoothed over time."""
    n = len(fr)
    raw = np.full(n, np.nan)
    for k in range(n):
        rows = _rows(fr[k]["lane"])
        if rows is None or np.isnan(off[k, 1]):
            continue
        far = rows[1]
        hip = _hip(trk.kpts[k])
        u_ref = far.u(_row_at_x(far, hip[0] if hip is not None else 960.0))[0]
        raw[k] = yard_theta(fr[k]["lane"], far, u_ref)
    return raw, _robust_smooth(raw)


def yard_line_positions(m: dict) -> list[tuple[int, float]]:
    """(frame, s from the start line) of every painted yard line crossing the far row."""
    out = []
    for k, R in enumerate(m["rails"]):
        if R is None:
            continue
        for u, _ in yard_line_hits(m["fr"][k]["lane"], R.far):
            out.append((k, float(R.far.s_of_u(u) + R.of - m["s_start"])))
    return out


def measure(run_id: str, k0: float) -> dict:
    fr = load(run_id)
    n = len(fr)
    off = unwrap(fr)
    trk = track_runner(fr, k0)
    off = fix_far(fr, off, trk, int(k0) - 5)
    s_start, support = start_line(fr, off, k0, trk)
    th_raw, th = yard_rungs(fr, off, trk)
    R = [None] * n
    if not np.isnan(s_start):
        for k in range(n):
            if not np.isnan(th[k]):
                R[k] = rails(fr[k], off[k], th[k])
    # Lateral line of the run from the ankles over the whole run (runners hold a line).
    w_ank = [R[k].s_through(trk.kpts[k, j, :2])[1] for k in range(n) if R[k] is not None
             for j in (L_ANK, R_ANK) if trk.kpts[k, j, 2] > KP_MIN]
    w_run = float(np.nanmedian(w_ank)) if w_ank else np.nan
    s = np.full(n, np.nan)
    for k in range(n):
        hip = _hip(trk.kpts[k])
        if R[k] is not None and hip is not None and not np.isnan(w_run):
            s[k] = R[k].s_below(hip, w_run) - s_start
    # Single-frame outliers (a mis-fitted row) sit well off the smooth hip path.
    ok = ~np.isnan(s)
    if ok.sum() > 7:
        idx = np.arange(n)
        filled = np.interp(idx, idx[ok], s[ok])
        med = np.array([np.median(filled[max(0, i - 3):i + 4]) for i in range(n)])
        s[ok & (np.abs(s - med) > SPIKE)] = np.nan
    return dict(fr=fr, off=off, s_start=s_start, start_support=support,
                track=trk, rails=R, w_run=w_run, s=s, rung_theta=th, rung_theta_raw=th_raw)
