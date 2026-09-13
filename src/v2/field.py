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


def markings(frame, turf, max_width=70):
    """Thin white structures lying on turf: the painted lines.

    Width is what separates them from the runway, which is also white and also
    touches turf but is an order of magnitude wider.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    near = cv2.dilate(turf, np.ones((41, 41), np.uint8))
    white = ((s < 85) & (v > 150) & (near > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(white)
    out = np.zeros_like(white)
    for j in range(1, n):
        w, h = st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]
        if min(w, h) <= max_width:
            out |= (lab == j).astype(np.uint8)
    return out


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
