"""Painted 5-yard lines, found by vanishing point and read on the lane centreline."""

from __future__ import annotations

import cv2
import numpy as np

FIELD_H = 790


def _h(p):
    return np.array([p[0], p[1], 1.0])


def _line(p, q):
    return np.cross(_h(p), _h(q))


def _meet(l1, l2):
    x = np.cross(l1, l2)
    if abs(x[2]) < 1e-9:
        return None
    return x[:2] / x[2]


def _white_on_turf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    turf = ((h > 28) & (h < 100) & (s > 45)).astype(np.uint8)
    turf = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((91, 91), np.uint8))
    white = ((s < 90) & (v > 145) & (turf > 0)).astype(np.uint8)
    return cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def segments(frame, lane_mask, min_len=90):
    band = _white_on_turf(frame)
    band[cv2.dilate(lane_mask, np.ones((9, 9), np.uint8)) > 0] = 0
    segs = cv2.HoughLinesP(band, 1, np.pi / 720, threshold=40,
                           minLineLength=min_len, maxLineGap=40)
    if segs is None:
        return []
    out = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        p, q = np.array([x1, y1], float), np.array([x2, y2], float)
        L = float(np.linalg.norm(q - p))
        if L >= min_len:
            out.append((p, q, L, _line(p, q)))
    return out


def _vp_ransac(segs, iters=600, tol=0.012, rng=None):
    """Vanishing point supported by the most segments.

    Support is angular: a segment votes if the direction from its midpoint to the
    candidate point matches its own direction. Distance to the point cannot be
    used, because a vanishing point is often far outside the image.
    """
    if len(segs) < 3:
        return None, np.zeros(len(segs), bool)
    rng = rng or np.random.default_rng(0)
    best = (0, None, np.zeros(len(segs), bool))
    mids = np.array([(s[0] + s[1]) / 2 for s in segs])
    dirs = np.array([(s[1] - s[0]) / s[2] for s in segs])
    for _ in range(iters):
        i, j = rng.choice(len(segs), 2, replace=False)
        v = _meet(segs[i][3], segs[j][3])
        if v is None:
            continue
        u = v - mids
        n = np.linalg.norm(u, axis=1, keepdims=True)
        u = u / np.maximum(n, 1e-9)
        cos = np.abs(np.sum(u * dirs, axis=1))
        inl = cos > 1 - tol
        w = sum(segs[k][2] for k in np.nonzero(inl)[0])
        if w > best[0]:
            best = (w, v, inl)
    return best[1], best[2]


def detect(frame, lane_mask, axis, min_support=2, max_align=0.992):
    """Return [(point, direction, support)] for painted lines crossing the lane.

    Two families of white lines share the turf: the lane edges, sideline and hash
    rows run with the lane, the yard lines run across it. They cannot be told
    apart by image angle - that changes with framing, and in tight shots the yard
    lines run close to the lane's own direction. They can be told apart by
    vanishing point, which is a property of the world rather than of the framing.
    """
    segs = segments(frame, lane_mask)
    if len(segs) < 4:
        return []
    rng = np.random.default_rng(0)
    v1, inl1 = _vp_ransac(segs, rng=rng)
    rest = [s for s, k in zip(segs, inl1) if not k]
    v2, inl2 = _vp_ransac(rest, rng=rng) if len(rest) >= 3 else (None, None)

    def alignment(members):
        if not members:
            return 9e9
        return float(np.median([abs(float((s[1] - s[0]) @ axis / s[2]))
                                for s in members]))

    cands = [(v1, [s for s, k in zip(segs, inl1) if k])]
    if v2 is not None:
        cands.append((v2, [s for s, k in zip(rest, inl2) if k]))
    # Both families can sit within 40 degrees of the lane in a tight framing, so
    # no absolute angle separates them; the yard lines are whichever family is
    # less aligned with the lane than the other.
    cands.sort(key=lambda c: alignment(c[1]))
    vp, members = cands[0]
    if vp is None or len(members) < min_support:
        return []

    centre, _ = lane_centreline(lane_mask, axis)
    out = []
    for p, q, L, _ in members:
        mid = (p + q) / 2
        d = vp - mid
        n = np.linalg.norm(d)
        # Near-parallel lines meet the centreline at an unstable point; reject on
        # that rather than on angle, which is what framing changes.
        if n < 1e-6 or abs(float(d @ axis / n)) > max_align:
            continue
        d = d / n
        X = cross_at(mid, d, centre, axis)
        if X is None:
            continue
        s = float(X @ axis)
        lim = 2.2 * frame.shape[1]
        if not (-lim < s < lim):
            continue
        out.append((mid, d, L, s, X))
    if not out:
        return []
    return _group(out)


def _group(lines, tol=200):
    """Merge in crossing space: two detections of one painted line meet the
    centreline at the same place, whatever their midpoints look like."""
    lines.sort(key=lambda c: c[3])
    out, cur = [], [lines[0]]
    for c in lines[1:]:
        if c[3] - cur[-1][3] < tol:
            cur.append(c)
        else:
            out.append(_merge(cur))
            cur = [c]
    out.append(_merge(cur))
    return out


def _merge(group):
    w = np.array([g[2] for g in group], float)
    pt = np.average([g[0] for g in group], axis=0, weights=w)
    d = np.average([g[1] for g in group], axis=0, weights=w)
    X = np.average([g[4] for g in group], axis=0, weights=w)
    return pt, d / max(np.linalg.norm(d), 1e-9), float(w.sum()), X


def lane_centreline(lane_mask, axis):
    """Mid-line of the running lane - where the athlete actually runs."""
    ys, xs = np.nonzero(lane_mask)
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    return c, axis


def cross_at(pt, direction, centre, axis):
    """Where a yard line meets the lane centreline."""
    return _meet(_line(pt, pt + direction * 100.0),
                 _line(centre, centre + axis * 100.0))
