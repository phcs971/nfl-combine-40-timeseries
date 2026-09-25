"""Decode each run clip once and cache everything measured per frame.

Per frame: all YOLO pose detections, the lane geometry, and two small signatures
used by QC - the panel logo patch and a colour histogram of the scene. Writes cache/frames/<run_id>.pkl.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from lane import detect
from panel import frames

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/frames"
MODEL = ROOT / "models/yolo11m-pose.pt"
LOGO = (60, 810, 210, 960)


def signatures(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = LOGO
    logo = cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.resize(img[:800], (240, 100)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).ravel()
    return logo, hist / hist.sum()


def extract(run: pd.Series, model: YOLO | None, prev: list[dict] | None = None) -> list[dict]:
    """With `prev`, only the lane is recomputed and the cached detections are kept."""
    video = ROOT / "video" / f"{run.video_id}.mp4"
    out = []
    for k, img in frames(video, ss=run.clip_ss, dur=run.clip_frames / (30000 / 1001)):
        if prev is not None:
            if k < len(prev):
                out.append({**prev[k], "lane": detect(img)})
            continue
        r = model.predict(img, imgsz=1280, device="mps", verbose=False, conf=0.2)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        kpts = r.keypoints.data.cpu().numpy() if len(r.boxes) else np.zeros((0, 17, 3))
        conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        # No masking of people: the start line sits under the runner's hands, and the
        # lattice fit already rejects limbs that happen to look like dashes.
        lane = detect(img)
        logo, hist = signatures(img)
        out.append(dict(k=k, boxes=boxes.astype(np.float32), conf=conf.astype(np.float32),
                        kpts=kpts.astype(np.float32), lane=lane, logo=logo, hist=hist.astype(np.float32)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma-separated run_ids")
    ap.add_argument("--videos", default="", help="comma-separated video_ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lane-only", action="store_true", help="recompute lane on cached runs")
    a = ap.parse_args()
    runs = pd.read_csv(ROOT / "data/runs_raw.csv")
    if a.runs:
        runs = runs[runs.run_id.isin(a.runs.split(","))]
    if a.videos:
        runs = runs[runs.video_id.isin(a.videos.split(","))]
    CACHE.mkdir(parents=True, exist_ok=True)
    model = None if a.lane_only else YOLO(str(MODEL))
    for run in runs.itertuples():
        path = CACHE / f"{run.run_id}.pkl"
        if a.lane_only:
            if not path.exists():
                continue
            # Local cache written by this script.
            with open(path, "rb") as f:
                data = extract(pd.Series(run._asdict()), None, pickle.load(f))
        elif path.exists() and not a.force:
            continue
        else:
            data = extract(pd.Series(run._asdict()), model)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(run.run_id, len(data), "frames", flush=True)


if __name__ == "__main__":
    main()
