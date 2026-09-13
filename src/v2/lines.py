"""Step 2 - the painted 5-yard lines."""

from __future__ import annotations

import cv2
import numpy as np

from field import turf_mask


def _white_in_field(frame, region):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    w = ((s < 85) & (v > 150) & (region > 0)).astype(np.uint8)
    return cv2.morphologyEx(w, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def candidates(frame, region, min_len=70, min_elong=4.0):
    """Elongated white strokes on the field: yard lines, hash rows, sideline."""
    w = _white_in_field(frame, region)
    n, lab, st, _ = cv2.connectedComponentsWithStats(w)
    out = []
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] < 220:
            continue
        pts = np.argwhere(lab == j)[:, ::-1].astype(np.float32)
        (cx, cy), (bw, bh), _ = cv2.minAreaRect(pts)
        L, S = max(bw, bh), max(min(bw, bh), 1e-6)
        if L < min_len or L / S < min_elong:
            continue
        mu = pts.mean(0)
        d = np.linalg.svd(pts - mu, full_matrices=False)[2][0]
        if d[1] < 0:
            d = -d
        out.append({"p": mu, "d": d, "len": float(L), "n": int(len(pts))})
    return out


def _vp(cands, iters=500, tol=0.010, seed=0):
    if len(cands) < 3:
        return None, np.zeros(len(cands), bool)
    rng = np.random.default_rng(seed)
    P = np.array([c["p"] for c in cands])
    D = np.array([c["d"] for c in cands])
    W = np.array([c["len"] for c in cands])
    best = (0.0, None, np.zeros(len(cands), bool))
    for _ in range(iters):
        i, j = rng.choice(len(cands), 2, replace=False)
        l1 = np.cross(np.r_[P[i], 1], np.r_[P[i] + D[i], 1])
        l2 = np.cross(np.r_[P[j], 1], np.r_[P[j] + D[j], 1])
        x = np.cross(l1, l2)
        if abs(x[2]) < 1e-9:
            continue
        vp = x[:2] / x[2]
        u = vp - P
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        inl = np.abs(np.sum(u * D, axis=1)) > 1 - tol
        s = float(W[inl].sum())
        if s > best[0]:
            best = (s, vp, inl)
    return best[1], best[2]


def detect(frame, region, runway_axis, max_align=0.80, min_len=150,
           min_members=1):
    """Yard lines, as [(point, unit direction, support)].

    Selected directly against the runway axis rather than by vanishing-point
    families: across-lines are only a handful of the candidates, so they never
    win a family of their own and get absorbed into the along-runway pencil.
    """
    cands = [c for c in candidates(frame, region)
             if c["len"] >= min_len
             and abs(float(c["d"] @ runway_axis)) <= max_align]
    if len(cands) < min_members:
        return []
    return merge_collinear(cands)


def merge_collinear(cands, perp_tol=26.0, dir_tol=0.985):
    """Fuse fragments of one painted line.

    A line is broken by the runway, by players and by its own wear, so it arrives
    as several strokes. They are the same line when each lies on the other's
    infinite line, which is a far better test than proximity along the axis -
    two genuinely different yard lines can project close together near the
    vanishing point.
    """
    groups = []
    for c in sorted(cands, key=lambda z: -z["len"]):
        placed = False
        for g in groups:
            if abs(float(c["d"] @ g["d"])) < dir_tol:
                continue
            off = c["p"] - g["p"]
            perp = abs(float(off @ np.array([-g["d"][1], g["d"][0]])))
            if perp < perp_tol:
                w = g["len"] + c["len"]
                g["p"] = (g["p"] * g["len"] + c["p"] * c["len"]) / w
                d = g["d"] * g["len"] + c["d"] * c["len"] * np.sign(float(c["d"] @ g["d"]))
                g["d"] = d / max(np.linalg.norm(d), 1e-9)
                g["len"] = w
                placed = True
                break
        if not placed:
            groups.append(dict(c))
    return [(g["p"], g["d"], g["len"]) for g in groups]
