"""Quality metrics and the accept/reject rule for one run."""

import cv2
import numpy as np
import pandas as pd

from features import YD_PER_PERIOD
from yardage import FPS, yard_line_positions

RULES = {
    "clock_resid_p95": ("<=", 0.025),
    "clock_coverage": (">=", 0.6),
    "bib_agree": (">=", 0.8),
    "panel_min_corr": (">=", 0.85),
    "max_cut": ("<=", 0.35),
    "turf_min": (">=", 0.18),
    "turf_median": (">=", 0.4),
    "lane_frac": (">=", 0.9),
    "track_frac": (">=", 0.9),
    "start_support": (">=", 5),
    "x_at_stop_err": ("<=", 1.5),
    "x_at_zero": ("between", (-2.2, 0.1)),
    "max_step_yd": ("<=", 0.8),
}


def window(k0: float, final_time: float, n: int) -> range:
    """Clip frames of the dataset: the clock window plus one frame either side, so the
    exact start and stop fall inside the data."""
    return range(max(0, int(np.ceil(k0)) - 1), min(n, int(np.floor(k0 + final_time * FPS)) + 2))


def turf(hist: np.ndarray) -> float:
    """Share of saturated green in the scene: a split screen or graphic shrinks it."""
    return float(hist.reshape(18, 8)[3:9, 2:].sum())


def logo_reference(frames_by_run: list[list[dict]]) -> np.ndarray:
    return np.median(np.stack([f["logo"] for fr in frames_by_run for f in fr]), 0).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel() - a.mean()
    b = b.astype(np.float32).ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _at(x: pd.Series, k: float, reach: int = 5) -> float:
    """Value at frame k, extrapolating linearly over at most `reach` frames past the data."""
    if len(x) < 4 or k < x.index.min() - reach or k > x.index.max() + reach:
        return np.nan
    if x.index.min() <= k <= x.index.max():
        return float(np.interp(k, x.index, x.values))
    end = x.iloc[-4:] if k > x.index.max() else x.iloc[:4]
    slope, icpt = np.polyfit(end.index, end.values, 1)
    return float(slope * k + icpt)


def metrics(run: pd.Series, m: dict, df: pd.DataFrame, logo_ref: np.ndarray) -> dict:
    fr = m["fr"]
    n = len(fr)
    k0 = run.k0
    kf = k0 + run.final_time * FPS
    # The dataset window; positions at the clock start and stop are read from a few frames either side.
    live = window(k0, run.final_time, n)
    around = range(max(0, int(k0) - 3), min(n, int(np.ceil(kf)) + 4))
    win = live

    cuts = [cv2.compareHist(fr[k - 1]["hist"], fr[k]["hist"], cv2.HISTCMP_BHATTACHARYYA)
            for k in win if k > 0]
    x = df.set_index("frame").x_yd
    ok = x.loc[list(around)].dropna()
    n_live = x.loc[list(live)].notna().sum()
    return dict(
        clock_resid_p95=run.clock_resid_p95,
        clock_coverage=run.n_clock / max(1.0, run.final_time * FPS),
        bib_agree=run.bib_agree,
        panel_min_corr=min(_corr(fr[k]["logo"], logo_ref) for k in win),
        max_cut=max(cuts) if cuts else np.nan,
        turf_min=min(turf(fr[k]["hist"]) for k in win),
        turf_median=float(np.median([turf(fr[k]["hist"]) for k in win])),
        lane_frac=np.mean([m["rails"][k] is not None for k in live]),
        track_frac=n_live / max(1, len(live)),
        start_support=m["start_support"],
        x_at_stop=_at(ok, kf),
        x_at_zero=_at(ok, k0),
        max_step_yd=float(np.abs(np.diff(ok.values) / np.diff(ok.index)).max()) if len(ok) > 1 else np.nan,
        yardline_resid=_yardline_resid(m),
    )


def _yardline_resid(m: dict) -> float:
    """Median distance (yd) from painted yard lines to the nearest 5-yd mark of our scale."""
    yd = np.array([s for _, s in yard_line_positions(m)]) * YD_PER_PERIOD
    yd = yd[(yd > 2) & (yd < 38)]
    return float(np.median(np.abs(yd - 5 * np.round(yd / 5)))) if len(yd) else np.nan


def verdict(mt: dict) -> tuple[bool, str]:
    mt = {**mt, "x_at_stop_err": abs(mt["x_at_stop"] - 40.0)}
    fails = []
    for key, (op, thr) in RULES.items():
        v = mt[key]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            fails.append(f"{key}=nan")
            continue
        good = (v <= thr if op == "<=" else v >= thr if op == ">=" else thr[0] <= v <= thr[1])
        if not good:
            fails.append(f"{key}={v:.3g}")
    return not fails, ";".join(fails)
