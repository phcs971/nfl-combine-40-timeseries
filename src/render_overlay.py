"""Render detector overlays for the labeller's toggle."""

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

import fieldlines as FL
from yardage import lane, FIELD_H
from detectors import _model, ANKLES

SKEL = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16)]
CYAN,TICK,GRN,LINE,POSE,CTR = ((255,210,60),(60,120,255),(80,255,140),
                               (255,90,220),(240,240,245),(60,230,255))


def pose(frame, centre=(0.25,0.85), min_foot_y=0.42):
    H,W = frame.shape[:2]
    res = _model().predict(frame, conf=0.35, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints)==0:
        return None
    best, score = None, -1.0
    for kp, box in zip(res.keypoints.data, res.boxes.data):
        k = kp.cpu().numpy()
        x1,y1,x2,y2 = box[:4].cpu().numpy()
        cx = (x1+x2)/2/W
        if not (centre[0] <= cx <= centre[1]):
            continue
        ank = [k[i] for i in ANKLES if k[i][2] > 0.3]
        if not ank or max(a[1] for a in ank)/H < min_foot_y:
            continue
        s = (y2-y1)*(1.0-abs(cx-0.55))
        if s > score:
            score, best = s, k
    return best


def draw(frame):
    v = frame.copy()
    L = lane(frame)
    if L is None:
        return v
    mask, mu, ax = L
    centre, a_ = FL.lane_centreline(mask, ax)
    cv2.line(v, tuple((centre - a_*2400).astype(int)),
             tuple((centre + a_*2400).astype(int)), (0,215,255), 2, cv2.LINE_AA)
    for pt, d, _, X in FL.detect(frame, mask, ax):
        cv2.line(v, tuple((X - d*900).astype(int)), tuple((X + d*900).astype(int)),
                 LINE, 3, cv2.LINE_AA)
        cv2.circle(v, tuple(X.astype(int)), 12, GRN, -1, cv2.LINE_AA)
    k = pose(frame)
    if k is not None:
        for a,b in SKEL:
            if k[a][2] > 0.3 and k[b][2] > 0.3:
                cv2.line(v, tuple(np.int32(k[a][:2])), tuple(np.int32(k[b][:2])),
                         POSE, 3, cv2.LINE_AA)
        for p in k:
            if p[2] > 0.3:
                cv2.circle(v, tuple(np.int32(p[:2])), 4, POSE, -1, cv2.LINE_AA)
        hips = [k[i] for i in (11,12) if k[i][2] > 0.3]
        if hips:
            c = np.mean([h[:2] for h in hips], axis=0)
            cv2.circle(v, tuple(np.int32(c)), 15, CTR, 4, cv2.LINE_AA)
            cv2.drawMarker(v, tuple(np.int32(c)), CTR, cv2.MARKER_CROSS, 34, 3)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    src = sorted((root/"frames").glob("*.jpg"))
    out = root/"overlay"; out.mkdir(exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for i,p in enumerate(src,1):
        im = cv2.imread(str(p))
        cv2.imwrite(str(out/p.name), draw(im), [cv2.IMWRITE_JPEG_QUALITY,85])
        if i % 40 == 0:
            print(f"  {i}/{len(src)}", file=sys.stderr)
    print(f"{len(src)} overlay frames -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
