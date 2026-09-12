"""Painted 5-yard lines, projected across the running lane."""

from __future__ import annotations

import cv2
import numpy as np

FIELD_H = 790


def _white_on_turf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    turf = ((h > 28) & (h < 100) & (s > 45)).astype(np.uint8)
    turf = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((91, 91), np.uint8))
    white = ((s < 90) & (v > 145) & (turf > 0)).astype(np.uint8)
    return cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def detect(frame, lane_mask, axis, min_len=170):
    """Return [(point, unit direction, support length)] for each painted line.

    Searched over open turf with the lane itself masked out. Restricting to the
    band above the lane loses them: in the tighter framings the visible yard lines
    sit beside the lane, not above it.
    """
    band = _white_on_turf(frame)
    band[cv2.dilate(lane_mask, np.ones((9, 9), np.uint8)) > 0] = 0

    segs = cv2.HoughLinesP(band, 1, np.pi / 720, threshold=50,
                           minLineLength=min_len, maxLineGap=40)
    if segs is None:
        return []

    cand = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        d = np.array([x2 - x1, y2 - y1], float)
        L = float(np.linalg.norm(d))
        if L < min_len:
            continue
        d /= L
        if abs(float(d @ axis)) > 0.55:
            continue
        if d[1] < 0:
            d = -d
        cand.append((np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]), d, L))
    if not cand:
        return []

    cand.sort(key=lambda c: float(c[0] @ axis))
    out, cur = [], [cand[0]]
    for c in cand[1:]:
        if float(c[0] @ axis) - float(cur[-1][0] @ axis) < 120:
            cur.append(c)
        else:
            out.append(_merge(cur))
            cur = [c]
    out.append(_merge(cur))
    return [o for o in out if o[2] >= min_len]


def _merge(group):
    w = np.array([g[2] for g in group], float)
    pt = np.average([g[0] for g in group], axis=0, weights=w)
    d = np.average([g[1] for g in group], axis=0, weights=w)
    return pt, d / max(np.linalg.norm(d), 1e-9), float(w.sum())


def project_to(point, direction, target_y):
    """Where a line reaches a given image row, following its own direction."""
    if abs(direction[1]) < 1e-6:
        return None
    k = (target_y - point[1]) / direction[1]
    return point + direction * k
