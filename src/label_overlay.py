"""Detector overlay for the hand labeller.

Permissive by design: every white-on-turf segment that runs across the lane is
drawn, not only the vanishing-point family. Precision is the labeller's job -
the overlay's job is to make sure the real yard line is among what is shown.
"""

from __future__ import annotations

import cv2
import numpy as np

import fieldlines as FL
from yardage import lane, FIELD_H

CAND = (150, 90, 140)
LINE = (255, 90, 220)
DOT = (80, 255, 140)
AXIS = (0, 215, 255)
POSE = (240, 240, 245)
HIP = (60, 230, 255)
FOOT = (60, 170, 255)

SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)]

_POSE_MODEL = None
_POSE_OFF = False


def _pose_model():
    global _POSE_MODEL, _POSE_OFF
    if _POSE_MODEL is None and not _POSE_OFF:
        try:
            from ultralytics import YOLO
            _POSE_MODEL = YOLO("yolo11n-pose.pt")
        except Exception:
            _POSE_OFF = True
    return _POSE_MODEL


def _athlete(frame, x_range=(0.20, 0.88), min_foot_y=0.42):
    m = _pose_model()
    if m is None:
        return None
    res = m.predict(frame, conf=0.35, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0:
        return None
    best, score = None, -1.0
    H, W = frame.shape[:2]
    for kp, box in zip(res.keypoints.data, res.boxes.data):
        k = kp.cpu().numpy()
        x1, y1, x2, y2 = box[:4].cpu().numpy()
        cx = (x1 + x2) / 2 / W
        if not (x_range[0] <= cx <= x_range[1]):
            continue
        ank = [k[i] for i in (15, 16) if k[i][2] > 0.3]
        if not ank or max(a[1] for a in ank) / H < min_foot_y:
            continue
        s = (y2 - y1) * (1.0 - abs(cx - 0.55))
        if s > score:
            score, best = s, k
    return best


def draw(frame, show_pose=True, max_align=0.985):
    v = frame.copy()
    f = frame[:FIELD_H]
    L = lane(f)
    if L is None:
        return v
    mask, _mu, ax = L
    centre, _ = FL.lane_centreline(mask, ax)
    cv2.line(v, tuple((centre - ax * 2400).astype(int)),
             tuple((centre + ax * 2400).astype(int)), AXIS, 2, cv2.LINE_AA)

    for p, q, ln, _h in FL.segments(f, mask):
        d = (q - p) / ln
        if abs(float(d @ ax)) > max_align:
            continue
        cv2.line(v, tuple((p - d * 500).astype(int)),
                 tuple((q + d * 500).astype(int)), CAND, 1, cv2.LINE_AA)

    for _pt, d, _w, X in FL.detect_full(f, mask, ax)[0]:
        cv2.line(v, tuple((X - d * 900).astype(int)),
                 tuple((X + d * 900).astype(int)), LINE, 3, cv2.LINE_AA)
        cv2.circle(v, tuple(X.astype(int)), 11, DOT, -1, cv2.LINE_AA)

    k = _athlete(frame) if show_pose else None
    if k is not None:
        for a, b in SKEL:
            if k[a][2] > 0.3 and k[b][2] > 0.3:
                cv2.line(v, tuple(np.int32(k[a][:2])), tuple(np.int32(k[b][:2])),
                         POSE, 3, cv2.LINE_AA)
        hips = [k[i][:2] for i in (11, 12) if k[i][2] > 0.3]
        if hips:
            c = np.mean(hips, axis=0)
            cv2.drawMarker(v, tuple(np.int32(c)), HIP, cv2.MARKER_CROSS, 34, 3)
        feet = [k[i][:2] for i in (15, 16) if k[i][2] > 0.3]
        if feet:
            c = np.mean(feet, axis=0)
            cv2.drawMarker(v, tuple(np.int32(c)), FOOT, cv2.MARKER_TILTED_CROSS,
                           26, 3)
    return v
