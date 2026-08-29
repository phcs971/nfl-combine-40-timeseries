"""Read the combine broadcast timing panel by glyph template matching.

The panel font is fixed, so digits are matched against prototypes harvested from
the feed rather than passed to a general OCR engine. Field roles are not hard-coded:
panel layout varies by position group, so the running clock is identified as the
field whose value advances between frames.
"""

import argparse
import io
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

GLYPH = (36, 24)
BAND = (800, 1000)
MIN_CORR = 0.85
PNG_SIG = b"\x89PNG\r\n\x1a\n"

# Prototypes are harvested by clustering, then labelled once by inspection.
# Several variants per digit survive on purpose: the feed anti-aliases the panel
# at differing subpixel offsets and a single mean template matches them worse.
CLUSTER_LABELS = "015751546239182"


def decode_stream(buf: bytes) -> list[np.ndarray]:
    out, i = [], 0
    while i >= 0:
        j = buf.find(PNG_SIG, i + 1)
        chunk = buf[i:] if j < 0 else buf[i:j]
        out.append(np.array(Image.open(io.BytesIO(chunk)).convert("RGB")))
        i = j
    return out


def band_frames(video: str, ss: float, dur: float, fps: str) -> list[np.ndarray]:
    y0, y1 = BAND
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{ss}", "-t", f"{dur}", "-i", video,
         "-vf", f"fps={fps},crop=1920:{y1 - y0}:0:{y0}",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True)
    if p.returncode != 0:
        raise SystemExit(p.stderr.decode()[-800:])
    return decode_stream(p.stdout)


def _norm(mask: np.ndarray) -> np.ndarray:
    im = Image.fromarray((mask * 255).astype(np.uint8)).resize(GLYPH[::-1])
    v = np.asarray(im, dtype=float).ravel() / 255.0
    return v - v.mean()


def extract_glyphs(band: np.ndarray) -> list[tuple[int, np.ndarray]]:
    white = (band > 195).all(axis=2)
    lab, _ = ndimage.label(white)
    out = []
    for sl in ndimage.find_objects(lab):
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if 38 < h < 52 and 8 < w < 40:
            out.append((sl[1].start, w, _norm(white[sl])))
    return sorted(out)


def build_templates(video: str, ss: float, dur: float, out_path: str) -> None:
    glyphs = [g for f in band_frames(video, ss, dur, "4") for g in extract_glyphs(f)]
    cents, counts = [], []
    for _, _, v in glyphs:
        sims = [float(v @ c / (np.linalg.norm(v) * np.linalg.norm(c) + 1e-9))
                for c in cents]
        if sims and max(sims) > 0.90:
            counts[int(np.argmax(sims))] += 1
        else:
            cents.append(v)
            counts.append(1)
    order = np.argsort(-np.array(counts))[:len(CLUSTER_LABELS)]
    np.savez_compressed(out_path,
                        templates=np.array([cents[i] for i in order]),
                        labels=np.array(list(CLUSTER_LABELS)))
    print(f"{len(glyphs)} glyphs -> {len(order)} prototypes -> {out_path}")


class PanelReader:
    def __init__(self, templates_path: str):
        z = np.load(templates_path)
        self.tpl = z["templates"]
        self.lab = z["labels"]
        self.tnorm = np.linalg.norm(self.tpl, axis=1)

    def _digit(self, v: np.ndarray) -> tuple[str, float]:
        s = self.tpl @ v / (self.tnorm * np.linalg.norm(v) + 1e-9)
        i = int(np.argmax(s))
        return str(self.lab[i]), float(s[i])

    def read(self, band: np.ndarray) -> list[tuple[float, float]]:
        """Return [(x_centre, value)] for each 3-digit d.dd field in the panel."""
        glyphs = extract_glyphs(band)
        fields, cur = [], []
        for x, w, v in glyphs:
            # Gap between glyph edges, not origins: the decimal point opens a ~30px
            # hole inside a value, while separate fields sit >150px apart.
            if cur and x - (cur[-1][0] + cur[-1][1]) > 60:
                fields.append(cur)
                cur = []
            cur.append((x, w, v))
        if cur:
            fields.append(cur)

        out = []
        for f in fields:
            if len(f) != 3:
                continue
            digits, confs = zip(*(self._digit(v) for _, _, v in f))
            if min(confs) < MIN_CORR:
                continue
            out.append((float(np.mean([x for x, _, _ in f])),
                        float(f"{digits[0]}.{digits[1]}{digits[2]}")))
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ss", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--fps", default="30000/1001")
    ap.add_argument("--templates", default="data/glyph_templates.npz")
    ap.add_argument("--build-templates", action="store_true")
    args = ap.parse_args()

    if args.build_templates:
        build_templates(args.video, args.ss, args.dur, args.templates)
        return 0

    reader = PanelReader(args.templates)
    frames = band_frames(args.video, args.ss, args.dur, args.fps)
    rows = [reader.read(f) for f in frames]

    xs = {}
    for r in rows:
        for x, v in r:
            xs.setdefault(round(x / 50) * 50, []).append(v)
    print(f"{len(frames)} frames, fields at x~{sorted(xs)}")

    dt = 1001 / 30000
    for col in sorted(xs):
        vals = xs[col]
        moving = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
        role = "CLOCK" if moving > 0.5 * len(vals) else "static"
        print(f"  x~{col:5d}  n={len(vals):3d}  {role:6}  "
              f"{min(vals):.2f}..{max(vals):.2f}")
        if role == "CLOCK":
            d = np.diff(vals)
            step = d[(d > 0) & (d < 0.2)]
            print(f"          step mean={step.mean():.4f}s expected={dt:.4f}s  "
                  f"sd={step.std():.4f}  bad_steps={int((d < 0).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
