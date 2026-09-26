"""Read the broadcast lower-third: timing fields (d.dd) and the athlete's bib.

Digits are matched against glyph prototypes harvested from the feed itself
(`harvest`), labelled once by eye from the sheet it writes (`label`). The prototypes
are crops of the broadcast graphic, so only the labels are committed
(data/templates/glyph_labels.json); `templates` rebuilds them from a local copy of
the videos.
"""

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "cache/templates/glyph_templates.npz"
LABELS = ROOT / "data/templates/glyph_labels.json"
# The harvest is deterministic for these videos and settings, so the committed labels apply.
HARVEST_VIDEOS = ["0r_usy_GIAI", "-zgIolzsJxY", "UcatZ1FYe5E", "g__zmhX838U"]
HARVEST_FPS = 1.0

W, H = 1920, 1080
BAND_Y0, BAND_Y1 = 800, 976
TIME_X = (655, 1330)
TIME_Y = (845, 915)
BIB_X = (260, 500)
BIB_Y = (915, 965)
CANVAS = 32


def frames(video: str | Path, ss: float = 0.0, dur: float | None = None,
           fps: float | None = None, crop: tuple[int, int, int, int] | None = None):
    """Yield (index, bgr) from ffmpeg; crop is (x, y, w, h). Index counts output frames."""
    vf = []
    if fps:
        vf.append(f"fps={fps}")
    x, y, w, h = crop or (0, 0, W, H)
    vf.append(f"crop={w}:{h}:{x}:{y}")
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(video), "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    n, size = 0, w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield n, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            n += 1
    finally:
        p.stdout.close()
        p.wait()


BAND = (0, BAND_Y0, W, BAND_Y1 - BAND_Y0)


def _vec(gray: np.ndarray) -> np.ndarray:
    """Aspect-preserving fit into a square canvas, zero-mean unit-norm."""
    h, w = gray.shape
    s = (CANVAS - 4) / max(h, w)
    g = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    c = np.zeros((CANVAS, CANVAS), np.float32)
    oy, ox = (CANVAS - g.shape[0]) // 2, (CANVAS - g.shape[1]) // 2
    c[oy:oy + g.shape[0], ox:ox + g.shape[1]] = g
    v = c.ravel() - c.mean()
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class Glyph:
    x: int
    w: int
    vec: np.ndarray


def _components(mask: np.ndarray, intensity: np.ndarray, hmin: int, hmax: int, x0: int) -> list[Glyph]:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if hmin <= h <= hmax and 4 <= w <= 45:
            g = np.where(lab[y:y + h, x:x + w] == i, intensity[y:y + h, x:x + w], 0).astype(np.float32)
            out.append(Glyph(int(x + x0), int(w), _vec(g)))
    return sorted(out, key=lambda g: g.x)


def _time_roi(band: np.ndarray) -> np.ndarray:
    y0, y1 = TIME_Y[0] - BAND_Y0, TIME_Y[1] - BAND_Y0
    return band[y0:y1, TIME_X[0]:TIME_X[1]].min(axis=2)


def time_glyphs(band: np.ndarray) -> list[Glyph]:
    lo = _time_roi(band)
    return _components(lo > 170, lo, 38, 55, TIME_X[0])


def decimal_points(band: np.ndarray) -> list[int]:
    """x centres of decimal-point blobs in the lower half of the digit row."""
    lo = _time_roi(band)
    n, _, st, cen = cv2.connectedComponentsWithStats((lo > 170).astype(np.uint8))
    return [int(cen[i][0]) + TIME_X[0] for i in range(1, n)
            if 6 <= st[i][2] <= 14 and 6 <= st[i][3] <= 14 and st[i][1] > lo.shape[0] // 2]


def bib_glyphs(band: np.ndarray) -> list[Glyph]:
    """Digits only: the position letters precede them across a word gap."""
    y0, y1 = BIB_Y[0] - BAND_Y0, BIB_Y[1] - BAND_Y0
    # Cyan text on a magenta ground: green alone separates them.
    g = band[y0:y1, BIB_X[0]:BIB_X[1], 1]
    gl = _components(g > 110, g, 18, 34, BIB_X[0])
    for i in range(1, len(gl)):
        if gl[i].x - (gl[i - 1].x + gl[i - 1].w) > 10:
            return gl[i:]
    return []


def position_patch(band: np.ndarray) -> np.ndarray | None:
    """The position code before the bib (e.g. "WR"), as a fixed-size normalised patch."""
    y0, y1 = BIB_Y[0] - BAND_Y0, BIB_Y[1] - BAND_Y0
    g = band[y0:y1, BIB_X[0]:BIB_X[1], 1]
    gl = _components(g > 110, g, 18, 34, 0)
    word = []
    for i, c in enumerate(gl):
        if i and c.x - (gl[i - 1].x + gl[i - 1].w) > 10:
            break
        word.append(c)
    if not word or len(word) == len(gl):
        return None
    x0, x1 = word[0].x, word[-1].x + word[-1].w
    ys = np.nonzero((g[:, x0:x1] > 110).any(1))[0]
    crop = g[ys.min():ys.max() + 1, x0:x1].astype(np.float32)
    v = cv2.resize(crop, (72, 24), interpolation=cv2.INTER_AREA).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def split_fields(gl: list[Glyph], gap: int = 80) -> list[list[Glyph]]:
    fields, cur = [], []
    for g in gl:
        if cur and g.x - (cur[-1].x + cur[-1].w) > gap:
            fields.append(cur)
            cur = []
        cur.append(g)
    if cur:
        fields.append(cur)
    return fields


class Reader:
    def __init__(self, path: Path = TEMPLATES):
        z = np.load(path)
        self.t_vec, self.t_lab = z["time_vec"], z["time_lab"]
        self.b_vec, self.b_lab = z["bib_vec"], z["bib_lab"]

    @staticmethod
    def _match(v, vecs, labs):
        s = vecs @ v
        i = int(np.argmax(s))
        return str(labs[i]), float(s[i])

    def times(self, band: np.ndarray, min_score: float = 0.8) -> list[tuple[int, float | None]]:
        """[(x_centre, value)] per d.dd field; value None when a glyph is unreadable."""
        out, dots = [], decimal_points(band)
        for f in split_fields(time_glyphs(band)):
            digs = [self._match(g.vec, self.t_vec, self.t_lab) for g in f]
            xc = (f[0].x + f[-1].x + f[-1].w) // 2
            # A d.dd field has its point between the first and second digit.
            ok = (len(digs) == 3 and all(d != "x" and s >= min_score for d, s in digs)
                  and any(f[0].x + f[0].w <= d <= f[1].x for d in dots))
            out.append((xc, int("".join(d for d, _ in digs)) / 100 if ok else None))
        return out

    def bib(self, band: np.ndarray, min_score: float = 0.8) -> str | None:
        digs = [self._match(g.vec, self.b_vec, self.b_lab) for g in bib_glyphs(band)]
        if not digs or any(d == "x" or s < min_score for d, s in digs):
            return None
        return "".join(d for d, _ in digs)


def _cluster(vecs: list[np.ndarray], thr: float) -> tuple[np.ndarray, np.ndarray]:
    cents, counts = [], []
    for v in vecs:
        if cents:
            s = np.array(cents) @ v
            i = int(np.argmax(s))
            if s[i] > thr:
                counts[i] += 1
                continue
        cents.append(v)
        counts.append(1)
    order = np.argsort(-np.array(counts))
    return np.array(cents)[order], np.array(counts)[order]


def _sheet(cents: np.ndarray, counts: np.ndarray, path: Path) -> None:
    tiles = []
    for i, (c, n) in enumerate(zip(cents, counts)):
        t = c.reshape(CANVAS, CANVAS)
        t = ((t - t.min()) / (np.ptp(t) + 1e-9) * 255).astype(np.uint8)
        t = cv2.resize(t, (96, 96), interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 0, 22, 2, 2, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(t, f"{i}:{n}", (4, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        tiles.append(t)
    cols = 12
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def harvest(videos: list[str], out_dir: Path, fps: float = 1.0, max_proto: int = 96) -> None:
    tv, bv = [], []
    for v in videos:
        for _, band in frames(v, fps=fps, crop=BAND):
            # Only frames showing the timing panel, so other graphics don't pollute the bib set.
            if not decimal_points(band):
                continue
            tv += [g.vec for g in time_glyphs(band)]
            bv += [g.vec for g in bib_glyphs(band)]
    out_dir.mkdir(parents=True, exist_ok=True)
    tc, tn = _cluster(tv, 0.93)
    bc, bn = _cluster(bv, 0.93)
    tc, tn, bc, bn = tc[:max_proto], tn[:max_proto], bc[:max_proto], bn[:max_proto]
    np.savez_compressed(out_dir / "protos.npz", time_vec=tc, time_n=tn, bib_vec=bc, bib_n=bn)
    _sheet(tc, tn, out_dir / "time_sheet.png")
    _sheet(bc, bn, out_dir / "bib_sheet.png")
    print(f"time: {len(tv)} glyphs -> {len(tc)} protos; bib: {len(bv)} glyphs -> {len(bc)} protos")


def label(proto_path: Path, labels_json: Path) -> None:
    """labels_json: {"time": [...], "bib": [...]}, one entry per prototype, "x" = not a digit.

    An entry may be several digits: small bib digits sometimes touch and form one component.
    """
    z = np.load(proto_path)
    lab = json.loads(labels_json.read_text())
    tl, bl = np.array(lab["time"]), np.array(lab["bib"])
    TEMPLATES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TEMPLATES, time_vec=z["time_vec"][:len(tl)], time_lab=tl,
                        bib_vec=z["bib_vec"][:len(bl)], bib_lab=bl)
    print(f"wrote {TEMPLATES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("videos", nargs="+")
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--fps", type=float, default=1.0)
    l = sub.add_parser("label")
    l.add_argument("protos", type=Path)
    l.add_argument("labels", type=Path)
    sub.add_parser("templates", help="harvest the fixed videos and apply the committed labels")
    a = ap.parse_args()
    if a.cmd == "harvest":
        harvest(a.videos, a.out, a.fps)
    elif a.cmd == "label":
        label(a.protos, a.labels)
    else:
        out = ROOT / "cache/glyphs"
        harvest([str(ROOT / "video" / f"{v}.mp4") for v in HARVEST_VIDEOS], out, HARVEST_FPS)
        label(out / "protos.npz", LABELS)
