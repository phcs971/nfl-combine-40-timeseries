"""Read the position code on the panel for every run (the group video can include
other positions, e.g. a QB in a WR session).

`harvest` clusters the code patches and writes a sheet to label once; `label` stores
the labelled prototypes; `templates` does both with the committed labels
(data/templates/position_labels.json); the default command classifies every run
into data/panel_positions.csv. The prototypes are crops of the broadcast graphic and
stay local.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from panel import BAND, frames, position_patch

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "cache/templates/position_templates.npz"
LABELS = ROOT / "data/templates/position_labels.json"


def run_patch(video: Path, t_origin: float) -> np.ndarray | None:
    vs = [position_patch(b) for _, b in frames(video, ss=t_origin + 0.5, dur=2.5, fps=1, crop=BAND)]
    vs = [v for v in vs if v is not None]
    if not vs:
        return None
    v = np.mean(vs, 0)
    return v / (np.linalg.norm(v) + 1e-9)


def all_patches() -> tuple[pd.DataFrame, np.ndarray]:
    runs = pd.read_csv(ROOT / "data/runs_raw.csv")
    pats, keep = [], []
    for r in runs.itertuples():
        v = run_patch(ROOT / "video" / f"{r.video_id}.mp4", r.t_origin)
        if v is not None:
            pats.append(v)
            keep.append(r.run_id)
    return runs.set_index("run_id").loc[keep].reset_index(), np.array(pats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="classify", choices=["harvest", "label", "templates", "classify"])
    ap.add_argument("--out", type=Path, default=ROOT / "cache/positions")
    ap.add_argument("--labels", type=Path, default=LABELS)
    a = ap.parse_args()
    if a.cmd in ("harvest", "templates"):
        runs, P = all_patches()
        cents, members = [], []
        for i, v in enumerate(P):
            s = np.array(cents) @ v if cents else np.array([])
            if len(s) and s.max() > 0.9:
                members[int(s.argmax())].append(i)
            else:
                cents.append(v)
                members.append([i])
        a.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out / "protos.npz", vec=np.array(cents))
        tiles = []
        for c, m in zip(cents, members):
            t = c.reshape(24, 72)
            t = ((t - t.min()) / (np.ptp(t) + 1e-9) * 255).astype(np.uint8)
            t = cv2.copyMakeBorder(cv2.resize(t, (216, 72)), 0, 22, 2, 2, cv2.BORDER_CONSTANT, value=0)
            cv2.putText(t, f"{len(tiles)}: n={len(m)} {runs.video_id[m[0]][:4]}", (4, 90), 0, 0.45, 255, 1)
            tiles.append(t)
        cv2.imwrite(str(a.out / "sheet.png"), np.vstack(tiles))
        print(f"{len(P)} runs -> {len(cents)} prototypes")
    if a.cmd in ("label", "templates"):
        z = np.load(a.out / "protos.npz")
        TEMPLATES.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(TEMPLATES, vec=z["vec"], code=np.array(json.loads(a.labels.read_text())))
    if a.cmd == "classify":
        z = np.load(TEMPLATES)
        runs, P = all_patches()
        s = P @ z["vec"].T
        runs["panel_pos"] = z["code"][s.argmax(1)]
        runs["panel_pos_score"] = s.max(1).round(3)
        runs[["run_id", "panel_pos", "panel_pos_score"]].to_csv(ROOT / "data/panel_positions.csv", index=False)
        print(runs.groupby(["video_id", "panel_pos"]).size().to_string())


if __name__ == "__main__":
    main()
