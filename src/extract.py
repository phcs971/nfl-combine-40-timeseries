"""Decode each run clip once and cache everything measured per frame.

Per frame: all YOLO pose detections, the lane geometry, and two small signatures
used by QC - the panel logo patch and a colour histogram of the scene. Writes cache/frames/<run_id>.pkl.
"""

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
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


def _lane_only(args) -> str:
    run, path = args
    # Local cache written by this script.
    with open(path, "rb") as f:
        prev = pickle.load(f)
    data = extract(run, None, prev)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return run.run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma-separated run_ids")
    ap.add_argument("--videos", default="", help="comma-separated video_ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lane-only", action="store_true", help="recompute lane on cached runs")
    ap.add_argument("--workers", type=int, default=1, help="processes for --lane-only")
    a = ap.parse_args()
    runs = pd.read_csv(ROOT / "data/runs_raw.csv")
    if a.runs:
        runs = runs[runs.run_id.isin(a.runs.split(","))]
    if a.videos:
        runs = runs[runs.video_id.isin(a.videos.split(","))]
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.lane_only:
        jobs = [(row, CACHE / f"{row.run_id}.pkl") for _, row in runs.iterrows()
                if (CACHE / f"{row.run_id}.pkl").exists()]
        with ProcessPoolExecutor(a.workers) as ex:
            for rid in ex.map(_lane_only, jobs):
                print(rid, "lane", flush=True)
        return
    model = YOLO(str(MODEL))
    for run in runs.itertuples():
        path = CACHE / f"{run.run_id}.pkl"
        if path.exists() and not a.force:
            continue
        data = extract(pd.Series(run._asdict()), model)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(run.run_id, len(data), "frames", flush=True)


if __name__ == "__main__":
    main()
