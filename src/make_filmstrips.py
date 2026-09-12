"""Render the pipeline as horizontal strips.

Every step shows the same four frames of one run, with each stage's findings drawn
on top of the previous stage's, so the reader watches one run accumulate meaning
rather than meeting six unrelated sets of pictures.
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, "src")

VID = "video/Sr-Q6UjJq6g.mp4"
ATHLETE = "Cameron Ball"
FH = 790
BG = (16, 18, 22)
CYAN, YEL, GRN, PNK = (214, 193, 53), (78, 193, 242), (141, 214, 88), (220, 120, 255)


def grab(t):
    cap = cv2.VideoCapture(VID)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def strip(images, cell_h=430, gap=14, caption=None, title_h=64):
    cells = []
    for im in images:
        s = cell_h / im.shape[0]
        cells.append(cv2.resize(im, (max(1, int(im.shape[1] * s)), cell_h)))
    W = sum(c.shape[1] for c in cells) + gap * (len(cells) - 1)
    top = title_h if caption else 0
    out = np.full((cell_h + top, W, 3), BG, np.uint8)
    x = 0
    for c in cells:
        out[top:top + cell_h, x:x + c.shape[1]] = c
        x += c.shape[1] + gap
    if caption:
        cv2.putText(out, caption, (4, int(top * 0.66)), cv2.FONT_HERSHEY_DUPLEX,
                    1.05, (242, 244, 248), 2, cv2.LINE_AA)
    return out


def tag(im, text, org, scale=1.0, color=(255, 255, 255), thick=2):
    """Draw a label on a plate.

    The strips are viewed scaled well down, so thin outlined text vanishes; a
    solid plate keeps every label legible at any width.
    """
    scale *= 2.0
    thick += 2
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    x, y = org
    pad = 12
    cv2.rectangle(im, (x - pad, y - th - pad), (x + tw + pad, y + base + pad // 2),
                  (14, 15, 18), -1)
    cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick,
                cv2.LINE_AA)


def save(name, im):
    p = f"frames/steps/{name}.jpg"
    cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 87])
    print(f"  {p}  {im.shape[1]}x{im.shape[0]}")


def build():
    from read_clock import PanelReader, extract_glyphs
    from read_bib import read_bib, roster
    from detectors import detect_mats, lane_axis_of, detect_athlete
    from crossings import gather, crossings as find_cross
    from fit_runs import model
    import pandas as pd

    reader = PanelReader("data/glyph_templates.npz")
    f = pd.read_csv("data/fits.csv", dtype={"video_id": str})
    cand = f[(f.video_id == "Sr-Q6UjJq6g") & (f.player_name == ATHLETE)
             & (f.quality == "ok")]
    # Take the run's start from its own fit. Every label on these strips is read
    # from the frame it sits on; none is asserted from a constant.
    r = cand.sort_values("t_zero").iloc[0]
    T_ZERO = float(r.t_zero)

    ev = find_cross(gather(VID, T_ZERO, T_ZERO + 6.6))[:4]
    times = list(ev)
    frames = [fr for fr in (grab(x) for x in times) if fr is not None]

    ros = roster(2026, ["DT"])
    by_bib = dict(zip(ros.bib, ros.player_name))

    # 1 — raw
    save("1_source", strip([f.copy() for f in frames], caption=(
        "1  SOURCE  -  one athlete inside an uncut session, four moments of his run")))

    # 2 — clock
    lay2 = []
    for t, fr in zip(times, frames):
        v = fr.copy()
        band = fr[800:1000]
        fields = reader.read(band)
        if fields:
            # The panel shows several numbers; the clock is the one reading the
            # elapsed time, not simply the rightmost.
            x, val = min(fields, key=lambda z: abs(z[1] - (t - T_ZERO)))
            cv2.rectangle(v, (int(x) - 110, 840), (int(x) + 110, 928), GRN, 5)
            tag(v, f"clock {val:.2f}", (int(x) - 150, 812), 0.9, GRN)
        tag(v, f"t = {t - T_ZERO:.2f}s", (30, 82), 1.15, GRN)
        lay2.append(v)
    save("2_clock", strip(lay2, caption=(
        "2  READ THE CLOCK  -  the panel supplies t=0 and the run's own time base")))

    # 3 — + bib
    lay3 = []
    for v in lay2:
        v = v.copy()
        bib = read_bib(v[800:1000], reader)
        cv2.rectangle(v, (256, 902), (404, 962), YEL, 5)
        if bib:
            tag(v, f"DL {bib} = {by_bib.get(bib, '?')}", (258, 1060), 0.85, YEL)
        lay3.append(v)
    save("3_bib", strip(lay3, caption=(
        "3  IDENTIFY  -  bib number indexes the alphabetical position-group roster")))

    # 4 — + detections
    lay4 = []
    for v in lay3:
        v = v.copy()
        la = lane_axis_of(v[:FH])
        if la:
            mu, d = la
            cv2.line(v, tuple((mu - d * 2400).astype(int)),
                     tuple((mu + d * 2400).astype(int)), CYAN, 3, cv2.LINE_AA)
        for c, w in detect_mats(v[:FH]):
            cv2.circle(v, (int(c[0]), int(c[1])), 34, YEL, 6)
        at = detect_athlete(v[:FH])
        if at:
            foot, hip = at
            cv2.circle(v, (int(foot[0]), int(foot[1])), 30, GRN, 7)
            if hip is not None:
                cv2.circle(v, (int(hip[0]), int(hip[1])), 18, PNK, 5)
        lay4.append(v)
    save("4_detect", strip(lay4, caption=(
        "4  DETECT  -  lane axis (blue), mats (yellow), hip (pink), ground contact (green)")))

    # 5 — + the crossing itself
    lay5 = []
    for k, (t, v) in enumerate(zip(times, lay4), start=1):
        v = v.copy()
        at = detect_athlete(v[:FH])
        mats = detect_mats(v[:FH])
        if at and mats:
            foot = at[0]
            c, _ = min(mats, key=lambda m: abs(m[0][0] - foot[0]))
            cv2.line(v, (int(foot[0]), int(foot[1])), (int(c[0]), int(c[1])),
                     (255, 255, 255), 3, cv2.LINE_AA)
            tag(v, f"mat {k}   {t - T_ZERO:.2f}s", (30, 250), 1.05, (255, 255, 255))
        lay5.append(v)
    save("5_crossings", strip(lay5, caption=(
        "5  CROSS  -  foot meets mat inside one frame, so the pan cannot bias it")))

    # 6 — + the fitted state at each frame
    vmax, tau = r.v_max_yd_s, r.tau_s
    lay6 = []
    for t, v in zip(times, lay5):
        v = v.copy()
        dt = t - T_ZERO
        x = model(dt, vmax, tau)
        vel = vmax * (1 - np.exp(-dt / tau)) * 0.9144
        tag(v, f"x = {x:5.1f} yd", (30, 390), 1.05, CYAN)
        tag(v, f"v = {vel:4.1f} m/s", (30, 500), 1.05, CYAN)
        lay6.append(v)
    save("6_fit", strip(lay6, caption=(
        f"6  FIT  -  {r.player_name}: v_max {r.v_max_m_s:.2f} m/s, tau {tau:.2f}s, "
        f"residual {r.resid_yd:.2f} yd")))
