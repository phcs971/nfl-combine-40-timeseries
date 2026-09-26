"""Figures and GIFs for the README walkthrough -> docs/img/.

Needs the local videos and caches. Broadcast frames are NFL footage: they are written
at reduced resolution and credited in the README.
"""

import subprocess
import warnings
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features import YD_PER_PERIOD  # noqa: E402
from overlay import BONES, draw  # noqa: E402
from panel import BAND_Y0, BIB_Y, TIME_Y, Reader, bib_glyphs, frames, time_glyphs  # noqa: E402
from yardage import FPS, _hip, measure, yard_line_positions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/img"
RUN = "BEFga72DF3U_011"
WIDTH = 960
CLASS_COLORS = {"SKILL": "#2a6fdb", "STRONG": "#e0892b", "LINEMAN": "#3a9d5d"}


def _runs() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data/runs_raw.csv").set_index("run_id")
    raw["run_id"] = raw.index
    return raw


def _frames_at(run: pd.Series, ks: list[int]) -> dict[int, np.ndarray]:
    want, got = set(ks), {}
    for k, img in frames(ROOT / "video" / f"{run.video_id}.mp4", ss=run.clip_ss, dur=run.clip_frames / FPS):
        if k in want:
            got[k] = img.copy()
    return got


def _save(img: np.ndarray, name: str, width: int = WIDTH) -> None:
    h = int(img.shape[0] * width / img.shape[1])
    cv2.imwrite(str(OUT / name), cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 82])


def _label(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (16 + 22 * len(text), 64), (0, 0, 0), -1)
    cv2.putText(img, text, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)


def broadcast(run: pd.Series, k: int) -> None:
    img = _frames_at(run, [k])[k]
    _save(img, "01_broadcast.jpg")


def clock(run: pd.Series, k: int) -> None:
    img = _frames_at(run, [k])[k]
    band = img[BAND_Y0:BAND_Y0 + 176].copy()
    reader = Reader()
    out = band.copy()
    for g in time_glyphs(band):
        cv2.rectangle(out, (g.x, TIME_Y[0] - BAND_Y0 + 8), (g.x + g.w, TIME_Y[1] - BAND_Y0 - 6), (0, 200, 255), 2)
    for g in bib_glyphs(band):
        cv2.rectangle(out, (g.x, BIB_Y[0] - BAND_Y0 + 6), (g.x + g.w, BIB_Y[1] - BAND_Y0 - 8), (0, 200, 255), 2)
    reads = ", ".join(f"{v:.2f}" for _, v in reader.times(band) if v is not None)
    cv2.putText(out, f"clock read: {reads}   bib read: {reader.bib(band)}", (1350, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    _save(out[:, 0:1920], "02_clock.jpg", 1200)


def segmentation(video: str, t1: float = 150.0) -> None:
    scan = pd.read_csv(ROOT / "cache/scan" / f"{video}.csv")
    runs = _runs()
    runs = runs[runs.video_id == video]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    s = scan[(scan.t <= t1) & scan.value.notna()]
    ax.scatter(s.t, s.value, s=4, color="#555")
    for r in runs.itertuples():
        if r.t_origin < t1:
            ax.axvspan(r.t_origin, r.t_origin + r.final_time, color="#2a6fdb", alpha=0.18)
            ax.text(r.t_origin, 4.9, f"{r.final_time:.2f}", fontsize=8, color="#2a6fdb")
    ax.set_xlabel("video time (s)")
    ax.set_ylabel("panel clock (s)")
    ax.set_title(f"Clock read at 4 fps; shaded = detected runs ({video}, first {t1:.0f} s)", fontsize=10)
    ax.set_xlim(0, t1)
    fig.tight_layout()
    fig.savefig(OUT / "03_segmentation.png", dpi=110)
    plt.close(fig)


def lane(run: pd.Series, m: dict, k: int) -> None:
    img = _frames_at(run, [k])[k]
    lf = m["fr"][k]["lane"]
    for x0, y0, x1, y1 in lf.yard_lines:
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 3)
    for r, (row, col) in enumerate(zip(lf.rows, [(0, 0, 255), (255, 0, 0)])):
        ss = np.linspace(row.idx.min() - 1, row.idx.max() + 1, 60)
        cv2.polylines(img, [row.point(ss).astype(np.int32)], False, col, 2)
        for i, p in zip(row.idx, row.pts):
            cv2.circle(img, (int(p[0]), int(p[1])), 8, col, 3)
            if r == 1:
                cv2.putText(img, str(int(i + m["off"][k, 1])), (int(p[0]) - 10, int(p[1]) - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    for b in lf.transverse:
        cv2.line(img, tuple(b.ends[0].astype(int)), tuple(b.ends[1].astype(int)), (0, 140, 255), 5)
    _label(img, "dash rows, yard lines, start line")
    _save(img, "04_lane.jpg")


def pose(run: pd.Series, m: dict, k: int) -> None:
    img = _frames_at(run, [k])[k]
    for b in m["fr"][k]["boxes"]:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 160, 160), 2)
    K = m["track"].kpts[k]
    for a, b in BONES:
        if K[a, 2] > 0.3 and K[b, 2] > 0.3:
            cv2.line(img, (int(K[a, 0]), int(K[a, 1])), (int(K[b, 0]), int(K[b, 1])), (0, 255, 0), 4)
    R, hip = m["rails"][k], _hip(K)
    if R is not None and hip is not None:
        g = R.point([m["s"][k] + m["s_start"]], m["w_run"])[0]
        cv2.line(img, (int(hip[0]), int(hip[1])), (int(g[0]), int(g[1])), (255, 0, 255), 3)
        cv2.circle(img, (int(g[0]), int(g[1])), 11, (255, 0, 255), -1)
    _label(img, "all people (grey), tracked runner, measured point")
    _save(img, "05_pose.jpg")


def measurement(run: pd.Series, m: dict, ks: list[int]) -> None:
    got = _frames_at(run, ks)
    tiles = [cv2.resize(draw(got[k], m, k, (k - run.k0) / FPS), (640, 360), interpolation=cv2.INTER_AREA)
             for k in ks]
    cv2.imwrite(str(OUT / "06_measurement.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])


def calibration(n_runs: int = 60) -> None:
    runs = pd.read_csv(ROOT / "data/runs.csv")
    runs = runs[runs.status == "ok"].groupby("video_id").head(n_runs // 11 + 1)
    s = []
    for r in runs.itertuples():
        s += [v for _, v in yard_line_positions(measure(r.run_id, r.k0))]
    s = np.array(s)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (scale, name) in zip(axes, [(2.0, "assumed 2.0 yd per dash"), (YD_PER_PERIOD, "fitted 1.8 m per dash")]):
        yd = s * scale
        keep = (yd > 2) & (yd < 41)
        yd = yd[keep]
        res = yd - 5 * np.round(yd / 5)
        ax.scatter(yd, res, s=2, alpha=0.08, color="#333")
        lines = 5 * np.round(yd / 5)
        med = pd.Series(res).groupby(lines).median()
        ax.plot(med.index, med.values, "o-", color="#d33", label="median per painted line")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("painted yard line position in our coordinate (yd)")
        ax.set_ylim(-1.5, 1.5)
    axes[0].set_ylabel("offset from the 5-yd mark (yd)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "07_calibration.png", dpi=110)
    plt.close(fig)


def series_plot(run_id: str = RUN) -> None:
    s = pd.read_parquet(ROOT / "data/series.parquet")
    s = s[s.run_id == run_id]
    cols = [("x_yd", "position (yd)"), ("v_yds", "velocity (yd/s)"), ("trunk_angle", "trunk angle (deg)"),
            ("hip_height_ratio", "hip height / leg"), ("knee_lead", "lead knee (deg)"), ("foot_contact", "foot contact")]
    fig, axes = plt.subplots(len(cols), 1, figsize=(9, 9), sharex=True)
    for ax, (c, lab) in zip(axes, cols):
        ax.plot(s.t_clock, s[c], color="#2a6fdb", lw=1.4)
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t_clock (s)")
    fig.suptitle(f"{run_id}: one row per video frame", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "08_series.png", dpi=100)
    plt.close(fig)


def classes() -> None:
    runs = pd.read_csv(ROOT / "data/runs.csv")
    runs = runs[runs.status == "ok"]
    d = pd.read_parquet(ROOT / "data/series_by_distance.parquet").merge(runs[["run_id", "cls"]], on="run_id")
    cols = [("t_clock", "time (s)"), ("v_yds", "velocity (yd/s)"), ("trunk_angle", "trunk angle (deg)"),
            ("hip_height_ratio", "hip height / leg"), ("step_len_yd", "step length (yd)"), ("knee_lead", "lead knee (deg)")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, (c, lab) in zip(axes.ravel(), cols):
        for cls, g in d.groupby("cls"):
            q = g.groupby("x_yd")[c].quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(q.index, q[0.5], color=CLASS_COLORS[cls], lw=2, label=f"{cls} ({g.run_id.nunique()} runs)")
            ax.fill_between(q.index, q[0.25], q[0.75], color=CLASS_COLORS[cls], alpha=0.15)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("distance (yd)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Class median and interquartile range along the run", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "09_classes.png", dpi=90)
    plt.close(fig)


def rejected(run_id: str = "LDqozfzjaNU_027") -> None:
    run = _runs().loc[run_id]
    k = int(run.k0 + 1.5 * FPS)
    img = _frames_at(run, [k])[k]
    _label(img, "rejected: split screen (turf < 18%)")
    _save(img, "10_rejected_split_screen.jpg")


def gif(run_id: str, name: str, width: int = 640, fps: int = 10) -> None:
    mp4 = ROOT / "frames/qc" / f"{run_id}.mp4"
    subprocess.run(["uv", "run", "python", str(ROOT / "src/overlay.py"), "--video", "--speed", "1", "--", run_id],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf",
                    f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4", str(OUT / name)], check=True)


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    run = _runs().loc[RUN]
    m = measure(RUN, run.k0)
    kf = run.k0 + run.final_time * FPS
    k_start, k_mid = int(np.ceil(run.k0)) - 1, int(run.k0 + 1.66 * FPS)
    broadcast(run, k_start)
    clock(run, k_mid)
    segmentation(run.video_id)
    lane(run, m, k_start)
    pose(run, m, k_mid)
    measurement(run, m, [k_start, k_mid, int(np.floor(kf)) + 1])
    calibration()
    series_plot()
    classes()
    rejected()
    gif(RUN, "run_wr.gif")
    gif("W23Nvg3rw6U_022", "run_ol.gif")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:32s} {f.stat().st_size / 1e6:5.2f} MB")


if __name__ == "__main__":
    main()
