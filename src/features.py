"""Per-frame series for one run: clock time, yardage and dimensionless pose channels.

Angles are measured in the image plane (the broadcast camera path is the same for
every run). Limbs are labelled lead/trail by position along the lane rather than
left/right, because the pose model swaps sides when the legs cross in a side view.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from yardage import FPS, KP_MIN, L_ANK, L_HIP, L_SHO, L_WRI, R_ANK, R_HIP, R_SHO, R_WRI

# The dashes are 1.8 m apart; calibrate.py fits 1.7995 m from the field's yard lines.
YD_PER_PERIOD = 1.8 / 0.9144
L_ELB, R_ELB, L_KNE, R_KNE = 7, 8, 13, 14
MAX_GAP = 4


def _fill(x: np.ndarray, max_gap: int = MAX_GAP) -> np.ndarray:
    """Linearly fill interior NaN runs up to max_gap frames."""
    x = x.copy()
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[ok], x[ok])
    run_start = None
    for i in range(len(x) + 1):
        if i < len(x) and not ok[i]:
            run_start = i if run_start is None else run_start
        elif run_start is not None:
            if run_start > 0 and i < len(x) and i - run_start <= max_gap:
                x[run_start:i] = filled[run_start:i]
            run_start = None
    return x


def _smooth(x: np.ndarray, win: int = 7, order: int = 2, deriv: int = 0) -> np.ndarray:
    """Savitzky-Golay over each contiguous non-NaN segment."""
    out = np.full_like(x, np.nan, dtype=float)
    ok = ~np.isnan(x)
    i = 0
    while i < len(x):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < len(x) and ok[j]:
            j += 1
        seg = x[i:j]
        if len(seg) >= win:
            out[i:j] = savgol_filter(seg, win, order, deriv=deriv, delta=1 / FPS)
        elif deriv == 0:
            out[i:j] = seg
        i = j
    return out


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b (degrees) between b->a and b->c, per row."""
    u, v = a - b, c - b
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _lean(top: np.ndarray, bottom: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """Angle of bottom->top from image-up, positive when tilted toward the run direction."""
    v = top - bottom
    up = np.array([0.0, -1.0])
    ang = np.degrees(np.arctan2(np.abs(v[:, 0] * up[1] - v[:, 1] * up[0]), v @ up))
    sign = np.sign((v * fwd).sum(1))
    return ang * np.where(sign == 0, 1, sign)


def series(m: dict, k0: float) -> pd.DataFrame:
    fr, trk, R = m["fr"], m["track"], m["rails"]
    n = len(fr)
    k = np.arange(n)
    t = (k - k0) / FPS

    x = _fill(m["s"] * YD_PER_PERIOD)
    x_s = _smooth(x, 9, 2)
    v = _smooth(x, 15, 3, deriv=1)
    a = _smooth(x, 21, 3, deriv=2)

    # Keypoints: mask low confidence, fill short gaps, smooth each coordinate.
    K = trk.kpts.copy()
    K[K[..., 2] < KP_MIN] = np.nan
    P = np.full((n, 17, 2), np.nan)
    for j in range(17):
        for c in range(2):
            P[:, j, c] = _smooth(_fill(K[:, j, c]), 7, 2)

    # Run direction in the image at the runner: image of +s through the hip.
    fwd = np.full((n, 2), np.nan)
    ank_s = np.full((n, 2), np.nan)
    ground_y = np.full(n, np.nan)
    for i in range(n):
        if R[i] is None or np.isnan(m["s"][i]):
            continue
        s_g = m["s"][i] + m["s_start"]
        p0, p1 = R[i].point([s_g, s_g + 0.25], m["w_run"])
        fwd[i] = (p1 - p0) / np.linalg.norm(p1 - p0)
        ground_y[i] = p0[1]
        # Raw keypoints here: a ground contact lasts ~3 frames and smoothing erases it.
        for side, j in enumerate((L_ANK, R_ANK)):
            if not np.isnan(K[i, j, :2]).any():
                ank_s[i, side] = R[i].s_through(K[i, j, :2])[0] * YD_PER_PERIOD

    sho = P[:, [L_SHO, R_SHO]].mean(1)
    hip = P[:, [L_HIP, R_HIP]].mean(1)
    trunk = _lean(sho, hip, fwd)

    # Lead leg = the one whose ankle is further along the lane.
    lead_is_left = ank_s[:, 0] >= ank_s[:, 1]
    lead = np.where(lead_is_left, 0, 1)
    sides = [(L_HIP, L_KNE, L_ANK), (R_HIP, R_KNE, R_ANK)]
    def leg(which):
        idx = np.where(which == 0, 0, 1)
        h = np.stack([P[i, sides[s][0]] for i, s in enumerate(idx)])
        kn = np.stack([P[i, sides[s][1]] for i, s in enumerate(idx)])
        an = np.stack([P[i, sides[s][2]] for i, s in enumerate(idx)])
        return h, kn, an
    h_l, k_l, a_l = leg(lead)
    h_t, k_t, a_t = leg(1 - lead)

    thigh = np.nanmedian(np.concatenate([np.linalg.norm(P[:, L_HIP] - P[:, L_KNE], axis=1),
                                         np.linalg.norm(P[:, R_HIP] - P[:, R_KNE], axis=1)]))
    shank = np.nanmedian(np.concatenate([np.linalg.norm(P[:, L_KNE] - P[:, L_ANK], axis=1),
                                         np.linalg.norm(P[:, R_KNE] - P[:, R_ANK], axis=1)]))
    leg_len = thigh + shank
    hip_height = (ground_y - hip[:, 1]) / leg_len

    elbow = np.nanmean(np.stack([_angle(P[:, L_SHO], P[:, L_ELB], P[:, L_WRI]),
                                 _angle(P[:, R_SHO], P[:, R_ELB], P[:, R_WRI])]), axis=0)
    arm_len = np.nanmedian(np.linalg.norm(P[:, L_SHO] - P[:, L_ELB], axis=1)
                           + np.linalg.norm(P[:, L_ELB] - P[:, L_WRI], axis=1))
    wr_fwd = np.stack([((P[:, j] - P[:, s]) * fwd).sum(1) for j, s in ((L_WRI, L_SHO), (R_WRI, R_SHO))], 1) / arm_len
    arm_swing = np.nanmax(wr_fwd, 1) - np.nanmin(wr_fwd, 1)

    # A planted foot is stationary on the ground plane.
    ank_v = np.stack([np.abs(np.gradient(_fill(ank_s[:, s], 2), 1 / FPS)) for s in range(2)], 1)
    slow = np.argmin(np.nan_to_num(ank_v, nan=np.inf), 1)
    thresh = np.maximum(0.35 * np.nan_to_num(v, nan=0.0), 1.5)
    contact = (np.nanmin(ank_v, 1) < thresh).astype(float)
    contact[np.isnan(ank_v).all(1)] = np.nan
    foot_pos = ank_s[np.arange(n), slow]
    # Steps are told apart by where the foot lands, which survives left/right swaps.
    touch = []
    # The set stance is not a step: stride channels start at the first touchdown after the clock starts.
    for i in range(int(np.ceil(k0)), n):
        if contact[i] != 1 or np.isnan(foot_pos[i]):
            continue
        if not touch and foot_pos[i] - np.nanmin(foot_pos[max(0, int(k0) - 5):int(k0) + 1], initial=np.inf) < 0.6:
            continue
        if touch and foot_pos[i] - touch[-1][1] < max(0.6, 0.12 * np.nan_to_num(v[i])):
            continue
        touch.append((i, foot_pos[i]))
    step_freq = np.full(n, np.nan)
    step_len = np.full(n, np.nan)
    for (i0, p0), (i1, p1) in zip(touch[:-1], touch[1:]):
        step_freq[i0:i1] = FPS / (i1 - i0)
        step_len[i0:i1] = p1 - p0

    df = pd.DataFrame(dict(
        frame=k, t_clock=np.round(t, 4),
        x_yd=np.round(x_s, 3), v_yds=np.round(v, 3), a_yds2=np.round(a, 3),
        trunk_angle=trunk,
        hip_height_ratio=hip_height,
        knee_lead=_angle(h_l, k_l, a_l), knee_trail=_angle(h_t, k_t, a_t),
        hip_flex_lead=_angle(sho, h_l, k_l), hip_flex_trail=_angle(sho, h_t, k_t),
        shin_angle_lead=_lean(k_l, a_l, fwd),
        thigh_sep=_angle(k_l, hip, k_t),
        elbow_angle=elbow,
        arm_swing=arm_swing,
        foot_contact=contact,
        step_freq_hz=step_freq,
        step_len_yd=step_len,
    ))
    for c in df.columns[5:]:
        if c not in ("foot_contact",):
            df[c] = df[c].round(3)
    df["valid"] = (~df[["x_yd", "trunk_angle", "hip_height_ratio"]].isna().any(axis=1)).astype(int)
    return df
