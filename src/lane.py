"""Per-frame geometry of the dash lane.

The lane is a white strip carrying two staggered rows of black dashes at a constant
world spacing (one "period"). In each frame the dashes of each row are indexed and a
1-D projective map index -> image position is fitted, which is exact for equally
spaced points on a line under perspective. Indices are only defined up to an integer
per row and frame; `yardage.py` resolves that across frames.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

PANEL_Y = 800


@dataclass
class Blob:
    c: np.ndarray       # centroid (x, y)
    ends: np.ndarray    # 2x2 endpoints along the long axis
    theta: float        # long-axis angle, degrees in [0, 180)
    length: float


@dataclass
class Row:
    p0: np.ndarray      # point on the row line
    d: np.ndarray       # unit direction, pointing to increasing index
    idx: np.ndarray     # integer index per dash (raw, arbitrary offset)
    pts: np.ndarray     # dash centres (n, 2)
    proj: np.ndarray    # (a, b, c): u = (a s + b) / (c s + 1)
    resid: float        # rms residual in periods

    def u(self, p: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(p) - self.p0) @ self.d

    def s_of_u(self, u):
        a, b, c = self.proj
        return (u - b) / (a - c * u)

    def u_of_s(self, s):
        a, b, c = self.proj
        return (a * s + b) / (c * s + 1)

    def point(self, s) -> np.ndarray:
        return self.p0 + np.outer(self.u_of_s(np.atleast_1d(s)), self.d)


@dataclass
class LaneFrame:
    rows: list[Row] = field(default_factory=list)   # [near, far] when both found
    transverse: list[Blob] = field(default_factory=list)
    lane_theta: float = np.nan


def _blobs(img: np.ndarray, exclude: list[tuple[int, int, int, int]] = ()) -> list[Blob]:
    hsv = cv2.cvtColor(img[:PANEL_Y], cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    white = (s < 60) & (v > 150)
    dark = (v < 90).astype(np.uint8)
    for x0, y0, x1, y1 in exclude:
        dark[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    k = np.ones((9, 9), np.uint8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 40 or area > 20000:
            continue
        sub = lab[y:y + h, x:x + w] == i
        ys, xs = np.nonzero(sub)
        pts = np.stack([xs + x, ys + y], 1).astype(np.float32)
        mean = pts.mean(0)
        ev, evec = np.linalg.eigh(np.cov((pts - mean).T))
        d = evec[:, 1]
        proj = (pts - mean) @ d
        length = float(np.ptp(proj))
        width = float(np.ptp((pts - mean) @ evec[:, 0])) + 1.0
        if length < 25 or length / width < 3.5:
            continue
        # Dashes and the start line are black on the white strip.
        pad = 5
        ya, yb = max(0, y - pad), min(white.shape[0], y + h + pad)
        xa, xb = max(0, x - pad), min(white.shape[1], x + w + pad)
        m = np.zeros((yb - ya, xb - xa), np.uint8)
        m[y - ya:y - ya + h, x - xa:x - xa + w] = sub
        ring = cv2.dilate(m, k) & ~m.astype(bool)
        if white[ya:yb, xa:xb][ring > 0].mean() < 0.45:
            continue
        theta = float(np.degrees(np.arctan2(d[1], d[0])) % 180)
        ends = np.stack([mean + proj.min() * d, mean + proj.max() * d])
        out.append(Blob(mean, ends, theta, length))
    return out


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _ransac_line(pts: np.ndarray, tol: float, iters: int = 200, rng=np.random.default_rng(0)):
    best = None
    n = len(pts)
    if n < 2:
        return None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        if np.linalg.norm(d) < 20:
            continue
        d = d / np.linalg.norm(d)
        nrm = np.array([-d[1], d[0]])
        inl = np.abs((pts - pts[i]) @ nrm) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    return best


def _fit_proj(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    A = np.stack([s, np.ones_like(s), -s * u], 1)
    return np.linalg.lstsq(A, u, rcond=None)[0]


def _index_row(pts: np.ndarray, blobs: list[Blob]) -> Row | None:
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    d = vt[0] if vt[0][0] > 0 else -vt[0]
    u = (pts - c) @ d
    order = np.argsort(u)
    u, pts, blobs = u[order], pts[order], [blobs[i] for i in order]
    # A dash cut by an occluder arrives as two blobs; merge pieces well inside one period.
    gaps = np.diff(u)
    if len(gaps) == 0:
        return None
    p = np.median(gaps)
    merged_u, merged_pts, lo, hi = [], [], [], []
    for i in range(len(u)):
        e = [(b - c) @ d for b in blobs[i].ends]
        if merged_u and u[i] - merged_u[-1] < 0.45 * p:
            lo[-1], hi[-1] = min(lo[-1], *e), max(hi[-1], *e)
            merged_u[-1] = (lo[-1] + hi[-1]) / 2
            merged_pts[-1] = c + merged_u[-1] * d
            continue
        lo.append(min(e))
        hi.append(max(e))
        merged_u.append(float(np.mean(e)))
        merged_pts.append(c + merged_u[-1] * d)
    u, pts = np.array(merged_u), np.array(merged_pts)
    if len(u) < 3:
        return None
    # Perspective makes the spacing drift smoothly; step indices by the local spacing.
    gaps = np.diff(u)
    base = np.median(gaps)
    inc = np.maximum(1, np.round(gaps / base)).astype(int)
    idx = np.concatenate([[0], np.cumsum(inc)])
    keep = np.ones(len(u), bool)
    for _ in range(3):
        proj = _fit_proj(idx[keep].astype(float), u[keep])
        a, b, cc = proj
        s_hat = (u - b) / (a - cc * u)
        # Re-index from the fit, then drop dashes that sit off the lattice (partly occluded).
        idx = np.round(s_hat).astype(int)
        res = s_hat - idx
        keep = np.abs(res) < 0.15
        if keep.sum() < 3 or len(np.unique(idx[keep])) < keep.sum():
            return None
    proj = _fit_proj(idx[keep].astype(float), u[keep])
    a, b, cc = proj
    s_hat = (u[keep] - b) / (a - cc * u[keep])
    resid = float(np.sqrt(np.mean((s_hat - idx[keep]) ** 2)))
    return Row(c, d, idx[keep] - idx[keep].min(), pts[keep], _fit_proj((idx[keep] - idx[keep].min()).astype(float), u[keep]), resid)


def detect(img: np.ndarray, exclude: list[tuple[int, int, int, int]] = ()) -> LaneFrame:
    blobs = _blobs(img, exclude)
    lf = LaneFrame()
    if len(blobs) < 4:
        return lf
    thetas = np.array([b.theta for b in blobs])
    # Dashes dominate the blob population; their shared angle is the lane direction.
    hist = [(sum(_ang_diff(t, t0) < 6 for t in thetas), t0) for t0 in thetas]
    lane_theta = max(hist)[1]
    along = [b for b in blobs if _ang_diff(b.theta, lane_theta) < 8]
    lf.lane_theta = float(np.median([b.theta for b in along]))
    lf.transverse = [b for b in blobs if _ang_diff(b.theta, lane_theta) > 25 and b.length > 60]
    pts = np.array([b.c for b in along])
    rows = []
    remaining = np.arange(len(along))
    for _ in range(2):
        if len(remaining) < 3:
            break
        inl = _ransac_line(pts[remaining], tol=7)
        if inl is None or inl.sum() < 3:
            break
        sel = remaining[inl]
        r = _index_row(pts[sel], [along[i] for i in sel])
        if r is not None:
            rows.append(r)
        remaining = remaining[~inl]
    if len(rows) == 2:
        # Near row = lower in the image (closer to the camera).
        mid = np.array([960.0, 0.0])
        ys = []
        for r in rows:
            t = (mid[0] - r.p0[0]) / r.d[0] if abs(r.d[0]) > 1e-6 else 0
            ys.append(r.p0[1] + t * r.d[1])
        rows = [rows[i] for i in np.argsort(ys)[::-1]]
    lf.rows = rows
    return lf


def intersect(p0, d0, p1, d1) -> np.ndarray | None:
    A = np.array([d0, -d1]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, p1 - p0)
    return p0 + t[0] * d0
