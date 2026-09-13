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


def candidates(frame, region, min_len=80, gap=45):
    """Straight white strokes on the field: yard lines, hash rows, sideline.

    Found by Hough rather than by connected components. A yard line is broken
    wherever a player, a bench or the runway crosses it, so as components its
    pieces fall under any useful length threshold and the line is lost.
    """
    w = _white_in_field(frame, region)
    segs = cv2.HoughLinesP(w, 1, np.pi / 720, threshold=45,
                           minLineLength=min_len, maxLineGap=gap)
    if segs is None:
        return []
    out = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        p, q = np.array([x1, y1], float), np.array([x2, y2], float)
        L = float(np.linalg.norm(q - p))
        if L < min_len:
            continue
        d = (q - p) / L
        if d[1] < 0:
            d = -d
        out.append({"p": (p + q) / 2, "d": d, "len": L, "n": int(L)})
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


def merge_collinear(cands, rho_tol=22.0, theta_tol=0.05):
    """Fuse detections of one painted line, in line space rather than by position.

    Hough returns many overlapping segments along a single line. Two segments are
    the same line when their normal form agrees, which is what (theta, rho) tests
    directly; comparing midpoints instead keeps duplicates whose midpoints happen
    to sit far apart along the line.
    """
    items = []
    for c in cands:
        th = float(np.arctan2(c["d"][1], c["d"][0])) % np.pi
        nrm = np.array([-np.sin(th), np.cos(th)])
        items.append((th, float(c["p"] @ nrm), c))

    groups = []
    for th, rho, c in sorted(items, key=lambda z: -z[2]["len"]):
        for g in groups:
            dth = abs(th - g["th"])
            dth = min(dth, np.pi - dth)
            if dth < theta_tol and abs(rho - g["rho"]) < rho_tol:
                w = g["len"] + c["len"]
                g["th"] = (g["th"] * g["len"] + th * c["len"]) / w
                g["rho"] = (g["rho"] * g["len"] + rho * c["len"]) / w
                g["p"] = (g["p"] * g["len"] + c["p"] * c["len"]) / w
                g["len"] = w
                break
        else:
            groups.append({"th": th, "rho": rho, "p": c["p"].copy(),
                           "len": c["len"]})
    out = []
    for g in groups:
        d = np.array([np.cos(g["th"]), np.sin(g["th"])])
        out.append((g["p"], d, g["len"]))
    return out
