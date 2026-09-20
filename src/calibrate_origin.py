"""Pin each extracted run's time origin to the panel clock.

`t_zero` from the run manifest is read at 4 fps and lands about 0.09 s before the
clock actually starts. Here the clock is read on every frame of the clip, which
locates its zero to a fraction of a frame.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image

from read_clock import PanelReader, BAND

FPS = 30000 / 1001


def clock_zero(frames: list[pathlib.Path], reader: PanelReader):
    y0, y1 = BAND
    fs, vs = [], []
    for p in frames:
        band = np.array(Image.open(p).convert("RGB"))[y0:y1]
        v = reader.read(band)
        if v:
            fs.append(int(p.stem[1:]))
            vs.append(max(v, key=lambda z: z[0])[1])
    if len(fs) < 30:
        return None, None
    fs, vs = np.array(fs, float), np.array(vs, float)
    run = (vs > 0.05) & (vs < vs.max() - 1e-9)
    if run.sum() < 20:
        return None, None
    # the clock advances at exactly the frame interval, so only the origin is
    # unknown; the median over frames shrugs off the occasional glyph misread
    f0 = float(np.median(fs[run] - vs[run] * FPS))
    resid = (fs[run] - f0) / FPS - vs[run]
    return f0, float(np.std(resid))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="label")
    ap.add_argument("--templates", default="data/glyph_templates.npz")
    ap.add_argument("--out", default="data/origins.csv")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    reader = PanelReader(a.templates)
    rows = []
    for d in sorted(pathlib.Path(a.root).iterdir()):
        meta = d / "meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        if "t_zero_panel" in m and not a.force:
            rows.append((d.name, m["athlete"], m["t_zero_panel"] - m["t_zero"],
                         m.get("origin_resid_s")))
            continue
        f0, resid = clock_zero(sorted((d / "frames").glob("*.jpg")), reader)
        if f0 is None:
            print(f"{d.name}: clock unreadable, keeping t_zero", file=sys.stderr)
            continue
        m["t_zero_panel"] = round(m["clip_start"] + (f0 - 1) / FPS, 4)
        m["origin_resid_s"] = round(resid, 4)
        meta.write_text(json.dumps(m, indent=1))
        rows.append((d.name, m["athlete"], m["t_zero_panel"] - m["t_zero"], resid))
        print(f"{d.name:22s} {m['athlete']:24s} shift {rows[-1][2]:+.3f}s "
              f"resid {resid:.4f}s")

    if rows:
        import csv
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "player_name", "shift_s", "resid_s"])
            w.writerows([(r[0], r[1], round(r[2], 4), r[3]) for r in rows])
        sh = np.array([r[2] for r in rows])
        print(f"\n{len(rows)} runs -> {a.out}   shift mean {sh.mean():+.3f} "
              f"sd {sh.std(ddof=1):.3f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
