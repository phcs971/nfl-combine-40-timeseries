"""Draw the measurement on video frames: lane rows with world indices, runner skeleton,
clock time and yardage. Writes frames/qc/<run_id>.jpg (a contact sheet), or with
--video an annotated clip with the per-frame series alongside (frames/qc/<run_id>.mp4)."""

import argparse
import subprocess
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from features import YD_PER_PERIOD, series
from panel import frames
from qc import window
from yardage import FPS, _hip, measure

ROOT = Path(__file__).resolve().parents[1]
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]


def draw(img: np.ndarray, m: dict, k: int, t: float) -> np.ndarray:
    out = img.copy()
    R = m["rails"][k]
    lf = m["fr"][k]["lane"]
    for r, (row, col) in enumerate(zip(lf.rows, [(0, 0, 255), (255, 0, 0)])):
        off = m["off"][k, r]
        for i, p in zip(row.idx, row.pts):
            cv2.circle(out, (int(p[0]), int(p[1])), 6, col, 2)
            if not np.isnan(off):
                cv2.putText(out, f"{int(i + off)}", (int(p[0]) - 8, int(p[1]) - 10), 0, 0.6, col, 2)
    if R is not None:
        # Yard ticks every 5 yd across the lane.
        for yd in range(0, 45, 5):
            a, b = R.point([yd / YD_PER_PERIOD + m["s_start"]] * 2, np.array([-0.5, 1.5]))
            if -500 < a[0] < 2500:
                cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 255), 2)
                cv2.putText(out, str(yd), (int(b[0]), int(b[1]) - 6), 0, 0.8, (0, 255, 255), 2)
    K = m["track"].kpts[k]
    if not np.isnan(K).any():
        for a, b in BONES:
            if K[a, 2] > 0.3 and K[b, 2] > 0.3:
                cv2.line(out, (int(K[a, 0]), int(K[a, 1])), (int(K[b, 0]), int(K[b, 1])), (0, 255, 0), 3)
    hip = _hip(K)
    if R is not None and hip is not None and not np.isnan(m["s"][k]):
        # The measured point: the ground below the hip, on the runner's line across the lane.
        g = R.point([m["s"][k] + m["s_start"]], m["w_run"])[0]
        cv2.line(out, (int(hip[0]), int(hip[1])), (int(g[0]), int(g[1])), (255, 0, 255), 2)
        cv2.circle(out, (int(g[0]), int(g[1])), 9, (255, 0, 255), -1)
    x = m["s"][k] * YD_PER_PERIOD
    label = f"t={t:+.2f}s  raw x={x:.2f}yd" if not np.isnan(x) else f"t={t:+.2f}s  raw x=nan"
    cv2.rectangle(out, (0, 0), (660, 60), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 42), 0, 1.3, (255, 255, 255), 3)
    return out


def sheet(run: pd.Series, ks: list[int], out_path: Path, scale: float = 0.4) -> None:
    warnings.filterwarnings("ignore")
    m = measure(run.run_id, run.k0)
    want = set(ks)
    tiles = {}
    for k, img in frames(ROOT / "video" / f"{run.video_id}.mp4", ss=run.clip_ss,
                         dur=run.clip_frames / FPS):
        if k in want:
            tiles[k] = cv2.resize(draw(img, m, k, (k - run.k0) / FPS), None, fx=scale, fy=scale)
    tiles = [tiles[k] for k in ks if k in tiles]
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


PANEL_W = 560
VALUES = ["x_yd", "v_yds", "a_yds2", "trunk_angle", "hip_height_ratio", "knee_lead", "knee_trail",
          "hip_flex_lead", "hip_flex_trail", "shin_angle_lead", "thigh_sep", "elbow_angle",
          "arm_swing", "foot_contact", "step_freq_hz", "step_len_yd"]
PLOTS = [("x_yd", (80, 200, 255)), ("v_yds", (80, 200, 255)), ("trunk_angle", (120, 230, 120)),
         ("hip_height_ratio", (120, 230, 120)), ("knee_lead", (230, 170, 90))]


def _text(img, s, xy, scale=0.5, col=(235, 235, 235), th=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def panel(df: pd.DataFrame, i: int, run: pd.Series, meta: dict, h: int, t_lo: float, t_hi: float) -> np.ndarray:
    p = np.full((h, PANEL_W, 3), 24, np.uint8)
    row = df.iloc[i]
    t = row.t_clock
    in_db = int(row.frame) in window(run.k0, run.final_time, 10**6)
    _text(p, f"{run.run_id}", (16, 30), 0.6, (255, 255, 255), 1)
    _text(p, f"{meta.get('position', '')} {meta.get('athlete', '')}  class {meta.get('cls', '')}  "
             f"split {meta.get('split', '')}", (16, 54), 0.5, (190, 190, 190))
    _text(p, f"frame {int(row.frame)}   t_clock {t:+.3f} s", (16, 88), 0.62, (255, 255, 255), 2)
    _text(p, "in dataset" if in_db else "outside dataset window", (380, 88), 0.45,
          (120, 230, 120) if in_db else (120, 120, 120))
    y = 118
    for j, c in enumerate(VALUES):
        v = row[c]
        s = "nan" if pd.isna(v) else (f"{v:.0f}" if c == "foot_contact" else f"{v:.2f}")
        x0 = 16 if j % 2 == 0 else 290
        _text(p, f"{c}", (x0, y), 0.45, (170, 170, 170))
        _text(p, s, (x0 + 170, y), 0.5, (255, 255, 255), 1)
        if j % 2:
            y += 22
    y += 12
    ph = (h - y - 10) // len(PLOTS)
    win = df[(df.t_clock >= t_lo) & (df.t_clock <= t_hi)]
    for c, col in PLOTS:
        vals = win[c]
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(lo) or hi - lo < 1e-9:
            y += ph
            continue
        x0, x1, y0, y1 = 16, PANEL_W - 16, y + 18, y + ph - 6
        cv2.rectangle(p, (x0, y0), (x1, y1), (60, 60, 60), 1)
        _text(p, f"{c}  [{lo:.1f}, {hi:.1f}]", (x0, y + 13), 0.42, (170, 170, 170))
        X = x0 + (win.t_clock.values - t_lo) / (t_hi - t_lo) * (x1 - x0)
        Y = y1 - (vals.values - lo) / (hi - lo) * (y1 - y0)
        ok = ~np.isnan(Y)
        pts = np.stack([X, Y], 1)
        done = win.t_clock.values <= t
        for mask, colr in ((ok, (90, 90, 90)), (ok & done, col)):
            seg = pts[mask].astype(np.int32)
            if len(seg) > 1:
                cv2.polylines(p, [seg], False, colr, 2 if colr == col else 1, cv2.LINE_AA)
        cx = int(x0 + (t - t_lo) / (t_hi - t_lo) * (x1 - x0))
        if x0 <= cx <= x1:
            cv2.line(p, (cx, y0), (cx, y1), (0, 200, 255), 1)
        y += ph
    return p


def video(run: pd.Series, out_path: Path, meta: dict, speed: float = 0.5, width: int = 1280,
          pad: float = 0.0) -> None:
    """The dataset window, optionally with `pad` seconds of context either side."""
    warnings.filterwarnings("ignore")
    m = measure(run.run_id, run.k0)
    df = series(m, run.k0)
    h = int(round(1080 * width / 1920 / 2) * 2)
    size = (width + PANEL_W, h)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
         "-r", f"{FPS * speed:.5f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "21", "-preset", "medium", str(out_path)], stdin=subprocess.PIPE)
    win = window(run.k0, run.final_time, 10**6)
    t_lo = (win.start - run.k0) / FPS - pad
    t_hi = (win.stop - 1 - run.k0) / FPS + pad
    for k, img in frames(ROOT / "video" / f"{run.video_id}.mp4", ss=run.clip_ss, dur=run.clip_frames / FPS):
        if k >= len(df):
            break
        if not t_lo - 1e-6 <= (k - run.k0) / FPS <= t_hi + 1e-6:
            continue
        left = cv2.resize(draw(img, m, k, (k - run.k0) / FPS), (width, h), interpolation=cv2.INTER_AREA)
        enc.stdin.write(np.hstack([left, panel(df, k, run, meta, h, t_lo, t_hi)]).tobytes())
    enc.stdin.close()
    enc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--frames", default="", help="comma-separated clip frame indices")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--video", action="store_true", help="annotated clip with the series panel")
    ap.add_argument("--speed", type=float, default=0.5, help="playback speed of the clip")
    ap.add_argument("--pad", type=float, default=0.0, help="seconds of context before the start and after the stop")
    a = ap.parse_args()
    run = pd.read_csv(ROOT / "data/runs_raw.csv").set_index("run_id").loc[a.run_id]
    run["run_id"] = a.run_id
    if a.video:
        info = pd.read_csv(ROOT / "data/runs.csv").set_index("run_id")
        meta = info.loc[a.run_id].to_dict() if a.run_id in info.index else {}
        video(run, a.out or ROOT / "frames/qc" / f"{a.run_id}.mp4", meta, a.speed, pad=a.pad)
        return
    if a.frames:
        ks = [int(v) for v in a.frames.split(",")]
    else:
        kf = run.k0 + run.final_time * FPS
        ks = [int(round(run.k0 + f * (kf - run.k0))) for f in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    sheet(run, ks, a.out or ROOT / "frames/qc" / f"{a.run_id}.jpg")


if __name__ == "__main__":
    main()
