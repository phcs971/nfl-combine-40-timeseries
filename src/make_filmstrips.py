"""Render each pipeline stage as a horizontal filmstrip."""

import sys

import cv2
import numpy as np

sys.path.insert(0, "src")

VID = "video/Sr-Q6UjJq6g.mp4"
FH = 790
PERF_BG = (18, 18, 20)


def grab(t, crop=None):
    cap = cv2.VideoCapture(VID)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    if crop:
        x, y, w, h = crop
        return fr[y:y + h, x:x + w]
    return fr[:FH]


def filmstrip(images, cell_h=280, gap=10, border=40, caption=None, title_h=54):
    """Lay images out horizontally inside a perforated film border."""
    cells = []
    for im in images:
        s = cell_h / im.shape[0]
        cells.append(cv2.resize(im, (max(1, int(im.shape[1] * s)), cell_h)))
    W = sum(c.shape[1] for c in cells) + gap * (len(cells) + 1)
    top = title_h if caption else 0
    H = cell_h + 2 * border + top
    strip = np.full((H, W, 3), PERF_BG, np.uint8)

    x = gap
    for c in cells:
        strip[top + border:top + border + cell_h, x:x + c.shape[1]] = c
        x += c.shape[1] + gap

    # Sprocket holes sit in the film borders, so the caption gets its own band
    # above them - printed over the perforations it is white on white.
    hw, hh, pitch = 26, 20, 62
    for cx in range(pitch // 2, W, pitch):
        for cy in (top + border // 2, H - border // 2):
            cv2.rectangle(strip, (cx - hw // 2, cy - hh // 2),
                          (cx + hw // 2, cy + hh // 2), (238, 238, 235), -1,
                          lineType=cv2.LINE_AA)
    if caption:
        cv2.rectangle(strip, (0, 0), (W, top), (10, 10, 11), -1)
        cv2.putText(strip, caption, (gap + 8, int(top * 0.68)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (245, 245, 240), 2,
                    cv2.LINE_AA)
    return strip


def around(frame, pt, w=1180, h=660):
    """Crop a window centred on a point, clamped to the frame."""
    H, W = frame.shape[:2]
    x = int(np.clip(pt[0] - w // 2, 0, max(0, W - w)))
    y = int(np.clip(pt[1] - h // 2, 0, max(0, H - h)))
    return frame[y:y + h, x:x + w]


def label(im, text, org=(14, 40), scale=0.95, color=(255, 255, 255)):
    im = im.copy()
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6,
                cv2.LINE_AA)
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2,
                cv2.LINE_AA)
    return im


def save(name, strip):
    path = f"frames/steps/{name}.jpg"
    cv2.imwrite(path, strip, [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(f"  {path}  {strip.shape[1]}x{strip.shape[0]}")


def step_clock():
    """The broadcast clock: t=0 and the run's own time base."""
    ims = []
    for t, tag in [(1.6, "0.00  set"), (2.5, "0.23  moving"), (3.6, "1.79  10yd"),
                   (5.0, "3.24"), (6.9, "5.11  stop")]:
        c = grab(t, crop=(300, 838, 1000, 120))
        if c is not None:
            ims.append(label(c, tag, (12, 34), 0.8, (120, 255, 170)))
    save("2_clock", filmstrip(ims, cell_h=118,
                              caption="2  READ THE CLOCK  -  t=0, splits, finish"))


def step_bib():
    """Bib number indexes the alphabetical position-group roster."""
    ims = []
    for t, tag in [(4.0, "DL 3  Barrett"), (88.0, "DL 8  Durant"),
                   (185.0, "DL 18  Keenan"), (250.0, "DL 25  Proctor")]:
        c = grab(t, crop=(255, 820, 330, 140))
        if c is not None:
            ims.append(label(c, tag, (10, 28), 0.62, (255, 220, 120)))
    save("3_bib", filmstrip(ims, cell_h=190,
                            caption="3  IDENTIFY THE ATHLETE  -  bib -> roster"))


def step_detect():
    from detectors import detect_mats, lane_axis_of, detect_athlete
    ims = []
    for t in (3.6, 4.5, 5.4, 6.3):
        f = grab(t)
        if f is None:
            continue
        v = f.copy()
        la = lane_axis_of(f)
        if la:
            mu, d = la
            a = (mu - d * 2400).astype(int)
            b = (mu + d * 2400).astype(int)
            cv2.line(v, tuple(a), tuple(b), (255, 210, 60), 3, cv2.LINE_AA)
        for c, w in detect_mats(f):
            cv2.circle(v, (int(c[0]), int(c[1])), 30, (60, 230, 255), 5)
        at = detect_athlete(f)
        if at:
            foot, hip = at
            cv2.circle(v, (int(foot[0]), int(foot[1])), 30, (90, 255, 120), 7)
            if hip is not None:
                cv2.circle(v, (int(hip[0]), int(hip[1])), 18, (255, 120, 220), 5)
            ims.append(around(v, foot))
        else:
            ims.append(v)
    save("4_detect", filmstrip(
        ims, caption="4  DETECT  -  lane axis (cyan), mats (yellow), foot (green)"))


def step_source():
    ims = []
    for t in (4.0, 60.0, 140.0, 250.0):
        f = grab(t)
        if f is not None:
            ims.append(f)
    save("1_source", filmstrip(
        ims, caption="1  SOURCE  -  one uncut position-group session, every athlete"))


def step_crossings():
    from crossings import gather, crossings as find
    from detectors import detect_mats, lane_axis_of, detect_athlete
    t0 = 1.77
    ev = find(gather(VID, t0, t0 + 6.6))
    ims = []
    for k, e in enumerate(ev[:4]):
        f = grab(e)
        if f is None:
            continue
        v = f.copy()
        at = detect_athlete(f)
        for c, w in detect_mats(f):
            cv2.circle(v, (int(c[0]), int(c[1])), 34, (60, 230, 255), 7)
        if at:
            foot = at[0]
            cv2.circle(v, (int(foot[0]), int(foot[1])), 30, (90, 255, 120), 7)
            # Label after cropping: the crop is centred on the foot, so a label
            # drawn on the full frame falls outside it.
            ims.append(label(around(v, foot),
                             f"mat {k + 1}    t = {e - t0:.2f}s", (18, 48), 1.15,
                             (120, 255, 170)))
    save("5_crossings", filmstrip(
        ims, caption="5  CROSSINGS  -  foot meets mat, one frame, no parallax"))


def step_fit():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    f = pd.read_csv("data/fits.csv")
    # Require a run that saw every mat: with only two crossings the fit is
    # exactly determined and its zero residual means nothing.
    f = f[(f.quality == "ok") & (f.n_crossings >= 4)]
    r = f.sort_values("resid_yd").iloc[0]
    vmax, tau = r.v_max_yd_s, r.tau_s
    t = np.linspace(0, r.final_clock, 300)
    x = vmax * (t + tau * np.exp(-t / tau) - tau)
    v = vmax * (1 - np.exp(-t / tau)) * 0.9144
    a = (vmax / tau) * np.exp(-t / tau) * 0.9144

    panels = []
    for ys, lab_, col in [(x, "position  (yd)", "#39d98a"),
                          (v, "velocity  (m/s)", "#4cc9f0"),
                          (a, "acceleration  (m/s2)", "#ff8fab")]:
        fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=150)
        fig.patch.set_facecolor("#0d0d0f")
        ax.set_facecolor("#0d0d0f")
        ax.plot(t, ys, color=col, lw=2.6)
        ax.set_xlabel("t (s)", color="#cfcfcf", fontsize=9)
        ax.set_title(lab_, color="#f2f2f2", fontsize=11, pad=8)
        for sp in ax.spines.values():
            sp.set_color("#3a3a3f")
        ax.tick_params(colors="#9a9aa0", labelsize=8)
        ax.grid(alpha=0.16, color="#8a8a90")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        panels.append(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
        plt.close(fig)
    save("6_fit", filmstrip(panels, cell_h=430, caption=(
        f"6  FIT  -  {r.player_name}: v_max {r.v_max_m_s:.2f} m/s, "
        f"tau {tau:.2f}s, residual {r.resid_yd:.2f} yd")))
