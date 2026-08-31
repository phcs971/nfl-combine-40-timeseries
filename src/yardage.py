"""Label the ground: bounding boxes for every distance mat and yard tick.

Given one frame, return the marks visible on the running lane with the yard each
one stands for. This replaces inferring a run's geometry from a fitted offset with
reading it off the ground directly.
"""

from __future__ import annotations

import numpy as np
import cv2

FIELD_H = 790


def lane(frame):
    """Lane mask, its centroid and unit direction along the run."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = ((s < 70) & (v > 110)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    mask = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(float)
    mu = pts.mean(0)
    d = np.linalg.svd(pts - mu, full_matrices=False)[2][0]
    return mask, mu, (d if d[0] >= 0 else -d)


def _rect(box):
    """Centre, long side, short side and unit long-axis of an oriented box.

    The long axis is taken from the corners rather than minAreaRect's angle:
    OpenCV's angle convention swaps which side it calls width, so deriving a
    direction from it points a mat across the lane about half the time.
    """
    box = np.asarray(box, float)
    edges = [(np.linalg.norm(box[(i + 1) % 4] - box[i]), box[(i + 1) % 4] - box[i])
             for i in range(4)]
    edges.sort(key=lambda e: -e[0])
    long_len, long_vec = edges[0]
    short_len = edges[2][0]
    return box.mean(0), long_len, max(short_len, 1e-6), long_vec / max(long_len, 1e-6)


def detect_mats(frame, axis, mask, cluster_px=190):
    """Distance mats: a dark face inside a yellow border, lying along the lane.

    Yellow alone is not enough - kit, shoes and painted field marks all pass it.
    A mat is additionally required to sit on the lane, to be dark inside, and to
    lie along the running direction, which is what separates it from a shirt in
    the crowd or a neon shoe crossing the frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    yel = ((h > 17) & (h < 36) & (s > 110) & (v > 110)).astype(np.uint8)
    yel = cv2.morphologyEx(yel, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    on_lane = cv2.dilate(mask, np.ones((35, 35), np.uint8))
    yel &= on_lane

    n, lab, st, ct = cv2.connectedComponentsWithStats(yel)
    parts = []
    for j in range(1, n):
        x, y, w, hh = st[j, :4]
        if st[j, cv2.CC_STAT_AREA] < 400 or w < 60:
            continue
        # The dark face is what separates a mat's border from the lane's painted
        # yellow lines, which are the same hue but sit on bare turf.
        if float((v[y:y + hh, x:x + w] < 100).mean()) < 0.15:
            continue
        parts.append((np.asarray(ct[j], float),
                      np.argwhere(lab == j)[:, ::-1].astype(np.float32)))

    used = [False] * len(parts)
    out = []
    for i, (c, pix) in enumerate(parts):
        if used[i]:
            continue
        grp, used[i] = [pix], True
        for k in range(i + 1, len(parts)):
            if used[k]:
                continue
            off = parts[k][0] - c
            along = abs(float(off @ axis))
            across = abs(float(off @ np.array([-axis[1], axis[0]])))
            # Border fragments of one mat sit close along the lane and closer
            # across it; a neighbouring mat is ten yards away.
            if along < cluster_px and across < 90:
                grp.append(parts[k][1])
                used[k] = True
        box = cv2.boxPoints(cv2.minAreaRect(np.vstack(grp)))
        ctr, w, hh, long_dir = _rect(box)
        # Only one of a mat's two border strips usually survives the dark test,
        # so accept a strip: its across-lane extent is irrelevant to the yard it
        # marks, which is read off its leading edge along the lane.
        if not (150 < w < 700 and 10 < hh < 190 and w / hh < 35):
            continue
        if abs(float(long_dir @ axis)) < 0.90:      # must run along the lane
            continue
        out.append(box)
    return out


def detect_ticks(frame, axis, mask, min_inliers=5):
    """The white 1-yard ticks painted on the turf alongside the lane.

    Not the black dashes on the lane surface - those are the run surface's own
    markings and do not correspond to yards. The ticks are collinear, so a RANSAC
    line rejects the painted field numbers and stray white pixels that survive a
    size filter, and the runner's side of the field picks which parallel row.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    green = ((h > 30) & (h < 95) & (s > 60)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(green)
    if n < 2:
        return []
    gm = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

    white = ((s < 75) & (v > 155) & (gm > 0)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n2, lab2, st2, c2 = cv2.connectedComponentsWithStats(white)
    cand = [(np.asarray(c2[j], float), j) for j in range(1, n2)
            if st2[j, cv2.CC_STAT_AREA] > 60
            and 10 < st2[j, 2] < 90 and 5 < st2[j, 3] < 50]
    if len(cand) < min_inliers:
        return []

    pts = np.array([c for c, _ in cand])
    lane_ctr = np.argwhere(mask > 0)[:, ::-1].mean(0)
    perp = np.array([-axis[1], axis[0]])
    best, rng = None, np.random.default_rng(0)
    for _ in range(300):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        if np.linalg.norm(d) < 40:
            continue
        d = d / np.linalg.norm(d)
        if abs(float(d @ axis)) < 0.9:          # ticks run along the lane
            continue
        nrm = np.array([-d[1], d[0]])
        inl = np.abs((pts - pts[i]) @ nrm) < 12
        if inl.sum() < min_inliers:
            continue
        # Prefer the row nearest the lane; the field carries several.
        score = inl.sum() - 0.02 * abs(float((lane_ctr - pts[i]) @ perp))
        if best is None or score > best[0]:
            best = (score, inl.copy())
    if best is None:
        return []

    out = []
    for keep, (c, j) in zip(best[1], cand):
        if not keep:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(
            np.argwhere(lab2 == j)[:, ::-1].astype(np.float32)))
        out.append((box, float(c @ axis), float(c @ perp)))
    return sorted(out, key=lambda z: z[1])


def row_spacing(ticks):
    """Median gap between consecutive ticks within each row."""
    if len(ticks) < 4:
        return float("nan")
    a = np.array([t[1] for t in ticks])
    p = np.array([t[2] for t in ticks])
    sp = []
    for sel in (p >= np.median(p), p < np.median(p)):
        r = np.sort(a[sel])
        if len(r) >= 3:
            sp.append(np.median(np.diff(r)))
    return float(np.median(sp)) if sp else float("nan")
