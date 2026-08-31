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


def detect_yard_lines(frame, axis, min_len=150):
    """The field's painted 5-yard lines, which cross the running direction.

    They are white on turf, so they fall outside a plain green mask - the mask has
    to be closed over them before they can be searched for inside it. They are
    then separated from the lane edges and sideline, which are long and white too,
    by running across the lane rather than along it.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    green = ((h > 30) & (h < 95) & (s > 60)).astype(np.uint8)
    field = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((81, 81), np.uint8))
    white = ((s < 85) & (v > 150) & (field > 0)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    segs = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=55,
                           minLineLength=min_len, maxLineGap=26)
    if segs is None:
        return []
    perp = np.array([-axis[1], axis[0]])
    cand = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        d = np.array([x2 - x1, y2 - y1], float)
        L = float(np.linalg.norm(d))
        if L < min_len:
            continue
        d /= L
        if abs(float(d @ axis)) > 0.72:      # runs along the lane: an edge, not a line
            continue
        cand.append((np.array([(x1 + x2) / 2, (y1 + y2) / 2]), d, L))
    if not cand:
        return []

    # Merge segments of one painted line by where it cuts the running axis.
    cand.sort(key=lambda c: float(c[0] @ axis))
    out, cur = [], [cand[0]]
    for c in cand[1:]:
        # A painted line is thick enough to return two parallel segments; five
        # yards is several times wider, so anything closer is the same line.
        if float(c[0] @ axis) - float(cur[-1][0] @ axis) < 220:
            cur.append(c)
        else:
            out.append(_merge_line(cur, axis))
            cur = [c]
    out.append(_merge_line(cur, axis))
    return out


def _merge_line(group, axis):
    w = np.array([g[2] for g in group])
    ctr = np.average([g[0] for g in group], axis=0, weights=w)
    d = np.average([g[1] * np.sign(g[1][1] or 1) for g in group], axis=0, weights=w)
    d = d / max(np.linalg.norm(d), 1e-6)
    return ctr, d, float(w.sum())


class YardTracker:
    """Carry tick identity across frames so a yard is a count, not a measurement.

    Spacing in pixels is not constant - it grows down the frame with perspective
    and changes as the camera moves - so any per-frame estimate of it is noisy and
    a missed tick doubles a gap. Counting is immune to all of that: once a tick
    keeps its identity between frames, its yard is its index, and pixel spacing is
    only ever needed to interpolate between two adjacent ticks.

    Indices are relative until something names a yard; a mat's leading edge is
    stationary in the world, so its index must not drift, which is the check.
    """

    def __init__(self, tol=0.45):
        self.tol = tol
        self.tracks = {}      # index -> last position along the axis
        self.next_hi = None
        self.next_lo = None

    @staticmethod
    def _spacing(pos):
        if len(pos) < 3:
            return float("nan")
        g = np.diff(np.sort(pos))
        g = g[g > 20]
        if not len(g):
            return float("nan")
        base = np.median(g)
        # Fold gaps that span a missed tick back onto the single-step spacing.
        folded = [x / max(1, round(x / base)) for x in g if x / base < 6]
        return float(np.median(folded)) if folded else float("nan")

    def update(self, ticks_along):
        pos = np.sort(np.asarray(ticks_along, float))
        sp = self._spacing(pos)
        if len(pos) < 2 or not np.isfinite(sp):
            return {}

        if not self.tracks:
            idx = np.round((pos - pos[0]) / sp).astype(int)
            self.tracks = {int(i): float(p) for i, p in zip(idx, pos)}
            return dict(self.tracks)

        prev_pos = np.array(list(self.tracks.values()))
        prev_idx = np.array(list(self.tracks.keys()))
        # Camera shift: the median move of marks that are fixed in the world.
        deltas = [p - prev_pos[np.argmin(np.abs(prev_pos - p))] for p in pos]
        shift = float(np.median(deltas))
        pred = prev_pos + shift

        assigned, used = {}, set()
        for p in pos:
            j = int(np.argmin(np.abs(pred - p)))
            if j not in used and abs(pred[j] - p) < self.tol * sp:
                assigned[int(prev_idx[j])] = float(p)
                used.add(j)
        # Ticks entering frame extend the index run by whole steps.
        if assigned:
            ref_i = min(assigned)
            ref_p = assigned[ref_i]
            for p in pos:
                if any(abs(p - q) < 1e-6 for q in assigned.values()):
                    continue
                k = ref_i + int(round((p - ref_p) / sp))
                if k not in assigned:
                    assigned[k] = float(p)
        self.tracks = assigned
        return dict(assigned)

    def index_of(self, along, spacing_hint=None):
        """Fractional index of an arbitrary position, e.g. a mat's leading edge."""
        if len(self.tracks) < 2:
            return float("nan")
        idx = np.array(list(self.tracks.keys()), float)
        pos = np.array(list(self.tracks.values()), float)
        order = np.argsort(pos)
        return float(np.interp(along, pos[order], idx[order]))
