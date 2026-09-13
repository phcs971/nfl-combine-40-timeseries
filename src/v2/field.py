"""Step 1 - the in-bounds playing field."""

from __future__ import annotations

import cv2
import numpy as np

FRAME_H = 790          # rows above the broadcast panel

H_LO, H_HI = 26, 52
S_LO = 45
V_LO, V_HI = 28, 185


def turf_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    return ((h >= H_LO) & (h <= H_HI) & (s >= S_LO) &
            (v >= V_LO) & (v <= V_HI)).astype(np.uint8)


def footprint(turf, min_piece=0.01):
    """Filled outline of the playing surface, runway included.

    The runway's middle is far from any turf pixel, so a proximity test around
    turf keeps only its edges; the footprint is what contains it.
    """
    m = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((121, 121), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] < min_piece * m.size:
            continue
        c, _ = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, c, -1, 1, -1)
    return out


def white_near_turf(frame, turf, reach=41):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return ((s < 88) & (v > 148) & (footprint(turf) > 0)).astype(np.uint8)


def split_white(frame, turf, wide=47, pad=21):
    """Separate wide white (the runway) from thin white (the painted lines).

    Done by morphological opening rather than by component width: a yard line
    that touches the runway is one connected component with it, so a per-component
    width test calls the line wide and discards it.
    """
    white = white_near_turf(frame, turf)
    core = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((wide, wide), np.uint8))
    broad = cv2.dilate(core, np.ones((pad, pad), np.uint8))
    thin = (white & ~broad.astype(bool)).astype(np.uint8)
    return broad.astype(np.uint8), thin


def markings(frame, turf):
    return split_white(frame, turf)[1]


def detect(frame, min_frac=0.04, min_piece=0.008):
    """Playing surface, closed just enough to contain the lines painted on it.

    Every sizeable turf region is kept, not only the largest: the runway splits
    the turf into separate pieces, and taking one of them drops half the field.
    The close bridges a painted line (~15 px) without reaching off the turf, so
    the lines end up inside the region while the stands stay outside.
    """
    turf = turf_mask(frame)
    if turf.mean() < min_frac:
        return None
    m = (turf | markings(frame, turf)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    keep = np.zeros_like(m)
    limit = min_piece * m.size
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] >= limit:
            keep |= (lab == j).astype(np.uint8)
    if keep.mean() < min_frac:
        return None
    return keep


def runway(frame, turf=None, min_frac=0.008):
    """The white running lane: mask, centre and unit direction along it.

    Found as the wide white structure touching turf - width is what distinguishes
    it from the painted lines, which are thin.
    """
    if turf is None:
        turf = turf_mask(frame)
    broad, _ = split_white(frame, turf)
    n, lab, st, _ = cv2.connectedComponentsWithStats(broad)
    if n < 2:
        return None
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[j, cv2.CC_STAT_AREA] < min_frac * frame.shape[0] * frame.shape[1]:
        return None
    m = (lab == j).astype(np.uint8)
    pts = np.argwhere(m)[:, ::-1].astype(float)
    c = pts.mean(0)
    d = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    return m, c, (d if d[0] >= 0 else -d)


def boundary(frame, min_frac=0.04, min_piece=0.004):
    """In-bounds region as a filled polygon, not a per-pixel colour mask.

    The playing surface is a rectangle in the world, so it is convex in the image.
    Taking the hull of the turf keeps what stands on the field - players, benches,
    the runway, the mats - inside the region. A colour mask instead punches a hole
    at every one of them, which fragments the painted lines that have to be found
    inside it.
    """
    turf = turf_mask(frame)
    if turf.mean() < min_frac:
        return None
    m = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((81, 81), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    pts = []
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] >= min_piece * m.size:
            pts.append(np.argwhere(lab == j)[:, ::-1])
    if not pts:
        return None
    hull = cv2.convexHull(np.vstack(pts).astype(np.int32))
    out = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out, hull
