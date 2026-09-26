"""Write datapackage.json: a Frictionless Data Package describing every data file,
with its size, SHA-256, row count and a typed, described schema for every column."""

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.1"
REPO = "https://github.com/phcs971/nfl-combine-40-timeseries"

CHANNELS = {
    "x_yd": ("yd", "position along the lane of the ground point under the hip centre, from the start line"),
    "v_yds": ("yd/s", "velocity: Savitzky-Golay derivative of x_yd (15 frames)"),
    "a_yds2": ("yd/s^2", "acceleration: Savitzky-Golay second derivative of x_yd (21 frames)"),
    "trunk_angle": ("deg", "shoulder-mid to hip-mid vs image vertical, positive leaning forward"),
    "hip_height_ratio": ("1", "hip height above the ground point under it / run-median leg length"),
    "knee_lead": ("deg", "knee angle of the leg further along the lane (180 = straight)"),
    "knee_trail": ("deg", "knee angle of the rear leg (180 = straight)"),
    "hip_flex_lead": ("deg", "trunk-thigh angle of the forward leg"),
    "hip_flex_trail": ("deg", "trunk-thigh angle of the rear leg"),
    "shin_angle_lead": ("deg", "forward leg's shin vs image vertical, positive leaning forward"),
    "thigh_sep": ("deg", "angle between the two thighs"),
    "elbow_angle": ("deg", "mean of both elbow angles"),
    "arm_swing": ("1", "fore-aft spread of the two wrists / arm length"),
    "foot_contact": ("0/1", "1 while a foot is planted (ankle stationary along the lane)"),
    "step_freq_hz": ("Hz", "1 / time between successive touchdowns, held until the next; empty before the first stride"),
    "step_len_yd": ("yd", "distance between successive touchdowns, held until the next; empty before the first stride"),
}

COLUMNS = {
    "run_id": ("", "run identifier, <video_id>_<nnn> in clock-start order within the video"),
    "video_id": ("", "YouTube video id (see videos.csv)"),
    "frame": ("", "frame index within the decoded clip of the run"),
    "t_clock": ("s", "time on the broadcast 40-yd clock at this frame (fitted, sub-frame zero)"),
    "phase": ("1", "fraction of the run: 0 at the clock start, 1 at the clock stop"),
    "valid": ("0/1", "1 where position and the core pose channels are all measured"),
    # runs.csv
    "group": ("", "session name from the video title"),
    "position": ("", "session position (WR, CB, SAF, RB, TE, LB, EDGE, OL, DT)"),
    "panel_pos": ("", "position code read from the broadcast panel (WR, DB, RB, TE, LB, DL, OL, QB)"),
    "cls": ("", "class label: SKILL, STRONG or LINEMAN"),
    "athlete": ("", "athlete key <panel_pos>-<bib>; bibs are unique within a position group"),
    "bib": ("", "bib number read from the broadcast panel (majority vote over the run)"),
    "attempt": ("", "attempt number of this athlete (1 or 2)"),
    "split": ("", "train or test; by athlete, stratified by position within class, seed 2026"),
    "status": ("", "ok, rejected (failed QC), excluded (not a class position) or duplicate"),
    "qc_reason": ("", "QC rules that failed, name=value, separated by ';'"),
    "time_40yd": ("s", "unofficial 40-yd time: the on-screen clock value when it stops"),
    "time_10yd": ("s", "unofficial 10-yd split shown on the panel (line groups only)"),
    "t_at_10yd": ("s", "clock time when the measured hip crosses 10 yd"),
    "clip_ss": ("s", "video time of the first decoded frame of the run's clip"),
    "clip_frames": ("", "number of decoded frames in the clip"),
    "k0": ("", "clip frame (fractional) where the clock reads 0"),
    "clock_resid_p95": ("s", "95th percentile |read - fit| of the clock over the run"),
    "clock_coverage": ("1", "share of run frames with a clock read"),
    "bib_agree": ("1", "share of bib reads agreeing with the majority"),
    "panel_min_corr": ("1", "lowest panel-logo correlation with the video's reference"),
    "max_cut": ("1", "largest frame-to-frame colour histogram change (Bhattacharyya)"),
    "turf_min": ("1", "lowest share of turf-green pixels in a frame"),
    "turf_median": ("1", "median share of turf-green pixels"),
    "lane_frac": ("1", "share of run frames with a lane measurement"),
    "track_frac": ("1", "share of run frames with the runner's position"),
    "start_support": ("", "frames supporting the start-line detection"),
    "x_at_zero": ("yd", "hip position at the clock start"),
    "x_at_stop": ("yd", "hip position at the clock stop"),
    "max_step_yd": ("yd", "largest frame-to-frame position change"),
    "yardline_resid": ("yd", "median distance of the painted yard lines from the 5-yd marks of our scale"),
    # runs_raw.csv
    "positions": ("", "session positions as listed in videos.csv"),
    "slot": ("px", "x centre of the panel field holding the live clock"),
    "t_origin": ("s", "video time of the clock start from the 4 fps scan"),
    "split10_coarse": ("s", "10-yd split from the 4 fps scan"),
    "final_time": ("s", "clock value when it stops (published as time_40yd)"),
    "split10": ("s", "10-yd split (published as time_10yd)"),
    "clock_monotone": ("", "clock reads never decrease over the run"),
    "n_clock": ("", "run frames with a clock read"),
    "panel_pos_score": ("1", "match score of the panel position code"),
    # videos.csv
    "season": ("", "combine year"),
    "title": ("", "YouTube title"),
    "channel": ("", "YouTube channel"),
    "upload_date": ("", "YouTube upload date"),
    "duration_s": ("s", "video duration"),
    "url": ("", "YouTube URL"),
    "width": ("px", "width of the analysed file"),
    "height": ("px", "height of the analysed file"),
    "fps": ("1/s", "frame rate of the analysed file"),
    "sha256": ("", "SHA-256 of the analysed video file"),
}
COLUMNS.update(CHANNELS)

RESOURCES = [
    ("runs", "data/runs.csv", "one row per detected run: identity, class, split, clock times, QC"),
    ("series", "data/series.parquet", "per-frame series of every kept run (primary product)"),
    ("series-csv", "data/series.csv", "CSV copy of series.parquet"),
    ("series-by-time", "data/series_by_time.parquet", "kept runs resampled to 101 points from clock start to stop"),
    ("series-by-distance", "data/series_by_distance.parquet", "kept runs resampled every 0.25 yd from 0 to 39 yd"),
    ("videos", "data/videos.csv", "source videos with provenance and file checksums"),
    ("runs-raw", "data/runs_raw.csv", "intermediate: runs found by the clock scan"),
    ("panel-positions", "data/panel_positions.csv", "intermediate: position code read per run"),
]


def _type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "number"
    return "string"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    resources = []
    for name, rel, desc in RESOURCES:
        path = ROOT / rel
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        fields = []
        for c in df.columns:
            unit, text = COLUMNS.get(c, ("", ""))
            if not text:
                raise SystemExit(f"{rel}: no description for column {c}")
            f = {"name": c, "type": _type(df[c]), "description": text}
            if unit:
                f["unit"] = unit
            fields.append(f)
        resources.append({
            "name": name, "path": rel, "description": desc,
            "format": path.suffix.lstrip("."),
            "mediatype": "text/csv" if path.suffix == ".csv" else "application/vnd.apache.parquet",
            "bytes": path.stat().st_size, "hash": f"sha256:{_sha256(path)}", "rows": len(df),
            "schema": {"fields": fields},
        })
    pkg = {
        "name": "nfl-combine-40-timeseries",
        "title": "NFL Combine 40-Yard Dash Movement Time Series (2026)",
        "version": VERSION,
        "created": date.today().isoformat(),
        "homepage": REPO,
        "description": "Per-frame position, velocity, acceleration and pose time series of 40-yard "
                       "dashes measured from the 2026 NFL Combine broadcasts, labelled SKILL, STRONG "
                       "or LINEMAN. See README.md and DATASHEET.md.",
        "licenses": [{"name": "CC-BY-4.0", "title": "Creative Commons Attribution 4.0",
                      "path": "https://creativecommons.org/licenses/by/4.0/"}],
        "contributors": [{"title": "Pedro Henrique Cordeiro Soares", "role": "author",
                          "email": "phcs.971@gmail.com", "path": "https://orcid.org/0009-0000-6041-2494"}],
        "sources": [{"title": "NFL YouTube channel, 2026 NFL Combine 40-yard dash videos (see data/videos.csv)",
                     "path": "https://www.youtube.com/@NFL"}],
        "keywords": ["time series classification", "sports biomechanics", "sprint", "pose estimation",
                     "NFL Combine", "40-yard dash"],
        "resources": resources,
    }
    (ROOT / "datapackage.json").write_text(json.dumps(pkg, indent=2) + "\n")
    print(f"datapackage.json: {len(resources)} resources")


if __name__ == "__main__":
    main()
