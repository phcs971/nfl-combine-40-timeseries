"""Two independent detectors for a combine run frame.

They are kept apart deliberately: the yard ticks and the athlete need opposite
image treatment - the ticks are small, static, high-frequency marks best found by
colour thresholding the field, while the athlete is a large deformable object that
needs a learned pose model. Sharing one pipeline compromised both.
"""

from __future__ import annotations

import numpy as np
import cv2

FIELD_H = 790


# --------------------------------------------------------------------------
# 1. yard ticks
# --------------------------------------------------------------------------
def detect_yard_ticks(frame: np.ndarray, min_inliers: int = 5,
                      near: np.ndarray | None = None):
    """Locate the sideline 1-yard ticks and the line they lie on.

    Returns (origin, direction, positions) with positions measured in pixels
    along the fitted line, or None. Ticks are collinear by construction, so a
    RANSAC line rejects the painted field numbers and stray white pixels that a
    size filter alone keeps.

    The field carries several parallel rows of white marks - the sideline ticks
    and the two inbound hash rows. Picking whichever row has most inliers makes
    the fit jump between them from frame to frame, and the apparent spacing jumps
    with it. `near` (the runner's foot) selects the row he is actually running
    beside.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))

    green = ((h > 30) & (h < 95) & (s > 60)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(green)
    if n < 2:
        return None
    gm = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

    white = ((s < 75) & (v > 155) & (gm > 0)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n2, lab2, st2, c2 = cv2.connectedComponentsWithStats(white)
    pts = np.array([c2[j] for j in range(1, n2)
                    if st2[j, cv2.CC_STAT_AREA] > 60
                    and 10 < st2[j, 2] < 90
                    and 5 < st2[j, 3] < 50])
    if len(pts) < min_inliers:
        return None

    best = None
    rng = np.random.default_rng(0)
    for _ in range(300):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        nrm = np.linalg.norm(d)
        if nrm < 40:
            continue
        d = d / nrm
        perp = np.array([-d[1], d[0]])
        off = np.abs((pts - pts[i]) @ perp)
        inl = off < 12
        if inl.sum() < min_inliers:
            continue
        score = float(inl.sum())
        if near is not None:
            score -= 0.02 * abs(float((np.asarray(near) - pts[i]) @ perp))
        if best is None or score > best[0]:
            best = (score, pts[i].copy(), d.copy(), inl.copy())
    if best is None:
        return None

    _, p0, d, inl = best
    sel = pts[inl]
    mu = sel.mean(0)
    d = np.linalg.svd(sel - mu, full_matrices=False)[2][0]
    if d[0] < 0:
        d = -d
    along = np.sort((sel - mu) @ d)
    return mu, d, along


def detect_mats(frame: np.ndarray):
    """Locate the numbered distance mats lying on the lane.

    Each has a saturated yellow border around a black face. The border alone also
    matches the lane's painted yellow lines, so a mat is required to be dark
    inside - that is what separates them (dark fraction ~0.2-0.4 against ~0.0).
    Returns a list of (centroid, width).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    yel = ((h > 18) & (h < 38) & (s > 120) & (v > 120)).astype(np.uint8)
    yel = cv2.morphologyEx(yel, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, ct = cv2.connectedComponentsWithStats(yel)
    out = []
    for j in range(1, n):
        x, y, w, hh = st[j, :4]
        if st[j, cv2.CC_STAT_AREA] < 900 or w < 90 or hh < 25:
            continue
        if float((v[y:y + hh, x:x + w] < 90).mean()) < 0.15:
            continue
        out.append((np.asarray(ct[j], float), int(w)))
    return out


def lane_axis_of(frame: np.ndarray):
    """Origin and unit direction of the running lane."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    lane = ((s < 70) & (v > 110)).astype(np.uint8)
    lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    lane = cv2.morphologyEx(lane, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(lane)
    if n < 2:
        return None
    m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(float)
    mu = pts.mean(0)
    d = np.linalg.svd(pts - mu, full_matrices=False)[2][0]
    return mu, (d if d[0] >= 0 else -d)


def tick_spacing(along: np.ndarray) -> float:
    """Median gap between consecutive ticks, robust to missed detections."""
    if len(along) < 3:
        return float("nan")
    g = np.diff(along)
    g = g[g > 5]
    if not len(g):
        return float("nan")
    base = np.median(g)
    # A missed tick doubles a gap; fold those back before taking the median.
    folded = [x / max(1, round(x / base)) for x in g]
    return float(np.median(folded))


# --------------------------------------------------------------------------
# 2. athlete
# --------------------------------------------------------------------------
_MODEL = None
ANKLES = (15, 16)   # COCO keypoints: left/right ankle


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO("yolo11n-pose.pt")
    return _MODEL


def detect_athlete(frame: np.ndarray, centre=(0.25, 0.85), conf=0.35,
                   min_foot_y: float = 0.42):
    """Return (ground_contact_xy, hip_xy) for the runner, or None.

    The tracking camera holds the runner near mid-frame, which is what separates
    him from officials and waiting athletes at the edges. Ground contact is the
    lower ankle: it shares the ground plane with the ticks, so its projection
    onto them carries no parallax term.
    """
    H, W = frame.shape[:2]
    res = _model().predict(frame, conf=conf, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0:
        return None

    best, best_score, best_foot = None, -1.0, None
    for kp, box in zip(res.keypoints.data, res.boxes.data):
        k = kp.cpu().numpy()
        x1, y1, x2, y2 = box[:4].cpu().numpy()
        cx = (x1 + x2) / 2 / W
        if not (centre[0] <= cx <= centre[1]):
            continue
        ank = [k[i] for i in ANKLES if k[i][2] > 0.3]
        if not ank:
            continue
        f = max(ank, key=lambda a: a[1])
        # The runner is on the lane in the lower half; officials and waiting
        # athletes stand higher in frame and were otherwise being picked.
        if f[1] / H < min_foot_y:
            continue
        score = (y2 - y1) * (1.0 - abs(cx - 0.55))
        if score > best_score:
            best_score, best, best_foot = score, k, f
    if best is None:
        return None
    foot = best_foot
    hips = [best[i] for i in (11, 12) if best[i][2] > 0.3]
    hip = np.mean([h[:2] for h in hips], axis=0) if hips else None
    return np.array(foot[:2]), hip


def project(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    return float((point - origin) @ direction)
