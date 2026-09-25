# NFL Combine 40-Yard Dash — Movement Time Series

Per-frame multivariate time series of every 40-yard dash in the 2026 NFL Combine
broadcasts, labelled with one of three position classes, for time-series
classification. Each run gives position along the lane, velocity, acceleration and
pose channels (trunk lean, hip height, joint angles, stride) at 29.97 Hz. Every
series spans the on-screen clock, from the frame just before it starts to the frame
just after it stops.

- 11 videos → 303 runs detected → **254 kept** (161 athletes), 47 rejected by
  quality control, 2 excluded (a QB).
- At least 24 athletes per class on each side of the train/test split.
- Distances are calibrated on the field's painted yard lines: they agree with our
  lane coordinate to a median 0.16 yd over the whole 40.
- Every value is measured from the video. No external timing data is used.

## Classes

| Class | Positions | Videos (`data/videos.csv`) |
|---|---|---|
| `SKILL` | WR, CB, SAF | WR G1 `0r_usy_GIAI`, WR G2 `BEFga72DF3U`, CB `-zgIolzsJxY`, SAF `UcatZ1FYe5E` |
| `STRONG` | RB, TE, LB, EDGE | RB `g__zmhX838U`, TE `9xc35LbeeUE`, LB `olzS7RidilY`, EDGE `LDqozfzjaNU` |
| `LINEMAN` | OL (OT, G, C), DT | OL G1 `W23Nvg3rw6U`, OL G2 `f9RAoGpGkIY`, DL `Sr-Q6UjJq6g` |

The class comes from the session video. QBs and specialists are excluded: the
panel's position code is read on every run, and one QB who ran inside the WR group 2
session is dropped.

## Dataset

### Counts

| Class | Athletes | Train: runs / athletes | Test: runs / athletes |
|---|---|---|---|
| `SKILL` | 58 | 45 / 29 | 47 / 29 |
| `STRONG` | 54 | 44 / 27 | 40 / 27 |
| `LINEMAN` | 49 | 39 / 25 | 39 / 24 |

Kept runs by position: WR 42, CB 32, SAF 18, RB 15, TE 29, LB 16, EDGE 24, OL 56,
DT 22. Most athletes run twice. The split is by athlete, so both attempts land on the
same side. It is stratified by position within each class and seeded
(`build_db.py`, `SEED = 2026`).

### Files

Every series covers the clock window plus one frame either side: the last frame
before the clock starts (t between −0.033 and 0 s) through the first frame after it
stops. The exact start and stop moments therefore fall inside the data.

| File | One row per | Contents |
|---|---|---|
| `data/runs.csv` | detected run (303) | identity, class, split, clock, QC metrics, `status` |
| `data/series.parquet` (+ `series.csv`) | video frame of a kept run (36k rows, 130–164 per run) | every channel on every frame |
| `data/series_by_time.parquet` | 1% step of a kept run (101 per run) | the clock window resampled from `phase` 0 (clock start) to 1 (clock stop); `t_clock` and `x_yd` keep the real time and distance |
| `data/series_by_distance.parquet` | 0.25-yd step of a kept run (157 per run) | the channels at each distance from 0 to 39 yd, with `t_clock` at each |

- **`series.parquet`** is the primary product.
- **`series_by_time.parquet`** gives every run the same length (101 points),
  which most time-series classifiers need.
- **`series_by_distance.parquet`** aligns runs by position on the field instead. It
  starts at the start line, which the hip reaches ~0.3 s after the clock starts.
  - It ends at 39 yd. The clock stops when any part of the athlete breaks the
    beam, while the hip is still ~0.4 yd short of 40.
  - 253 of the 254 runs span the full 0–39 yd.

In both resampled views, 0/1 channels (`foot_contact`) are resampled by nearest
frame, not interpolated. Nothing is extrapolated: the stride channels stay empty
before the first stride and after the last touchdown.

Intermediate files, also committed:
- `data/runs_raw.csv`: runs found by the clock scan.
- `data/panel_positions.csv`: position code per run.
- `data/glyph_templates.npz`, `data/position_templates.npz`: labelled prototypes
  for the panel reader.

### Loading

```python
import pandas as pd

runs = pd.read_csv("data/runs.csv")
runs = runs[runs.status == "ok"]
frames = pd.read_parquet("data/series.parquet").merge(runs[["run_id", "cls", "split"]], on="run_id")
grid = pd.read_parquet("data/series_by_time.parquet").merge(runs[["run_id", "cls", "split"]], on="run_id")

# (runs, 101 * channels) table for a classifier; reshape to (runs, 101, channels) if needed
channels = ["x_yd", "v_yds", "trunk_angle", "hip_height_ratio"]
X = grid.pivot(index="run_id", columns="phase", values=channels)
```

### Example: one run, per frame

`BEFga72DF3U_009`, a WR (`WR-39`), test split, clock 4.26 s:

| t_clock | x_yd | v_yds | a_yds2 | trunk_angle | hip_height_ratio | knee_lead | foot_contact | step_len_yd |
|---|---|---|---|---|---|---|---|---|
| −0.01 | −1.05 | 1.97 | 9.8 | 80.4 | 0.58 | 109.6 | 1 | – |
| 0.02 | −0.97 | 2.40 | 9.9 | 64.6 | 0.53 | 82.8 | 0 | – |
| 0.49 | 0.99 | 5.99 | 4.9 | 35.2 | 0.67 | 123.2 | 0 | 1.11 |
| 1.66 | 10.06 | 8.98 | 2.6 | 17.5 | 0.78 | 142.4 | 1 | 1.43 |
| 2.99 | 23.90 | 11.44 | 3.3 | 14.0 | 0.91 | 127.3 | 0 | 2.26 |
| 4.26 | 39.21 | 10.79 | −11.4 | 24.1 | 1.02 | 85.2 | 0 | – |
| 4.29 | 39.55 | 10.11 | −12.8 | 25.6 | 1.01 | 109.3 | 0 | – |

The first and last rows are the bracketing frames either side of the clock. The
clock starts on first movement. At that point the hip is still 1 yd behind the
start line, low (0.53 of leg length), with the trunk at 65°. By 10 yd the athlete
runs at 9 yd/s with 17° of lean, and by the stop they are upright at about 11 yd/s.
The step channels are empty until the first stride completes and after the last
touchdown.

### Channels

Every channel is computed on every frame. Pose channels are dimensionless (degrees
or ratios), so absolute body size is not encoded.

| Channel | Unit | Meaning |
|---|---|---|
| `frame`, `t_clock` | –, s | clip frame index; time on the broadcast clock (one frame before 0 → one frame after the final time) |
| `phase` | – | `series_by_time` only: fraction of the run, 0 at the clock start, 1 at the stop |
| `x_yd` | yd | position along the lane of the ground point under the hip, from the start line |
| `v_yds`, `a_yds2` | yd/s, yd/s² | velocity and acceleration, Savitzky–Golay derivatives of `x_yd` |
| `trunk_angle` | deg | shoulder-mid → hip-mid vs vertical, positive leaning forward |
| `hip_height_ratio` | – | hip height above the ground / leg length |
| `knee_lead`, `knee_trail` | deg | knee angle of the forward / rear leg (180 = straight) |
| `hip_flex_lead`, `hip_flex_trail` | deg | trunk–thigh angle per leg |
| `shin_angle_lead` | deg | forward leg's shin vs vertical |
| `thigh_sep` | deg | angle between the thighs |
| `elbow_angle` | deg | mean elbow angle |
| `arm_swing` | – | fore–aft spread of the wrists / arm length |
| `foot_contact` | 0/1 | a foot is planted (ankle stationary on the ground plane) |
| `step_freq_hz`, `step_len_yd` | Hz, yd | from successive touchdowns after the clock starts, held until the next one |
| `valid` | 0/1 | position and the core pose channels are all measured |

Legs are labelled lead/trail by position along the lane rather than left/right. The
pose model swaps sides when the legs cross in a side view.

### `runs.csv` columns

| Group | Columns |
|---|---|
| Identity | `run_id`, `video_id`, `group`, `position`, `panel_pos`, `cls`, `athlete` (`<panel code>-<bib>`), `bib`, `attempt`, `split` |
| Outcome | `status` (`ok` / `rejected` / `excluded` / `duplicate`), `qc_reason` |
| Clock | `final_time`, `split10` (line groups), `k0` (clip frame where the clock reads 0), `clip_ss`, `clip_frames` |
| Measured checks | `t_at_10yd` (hip crossing time), `x_at_zero`, `x_at_stop` (hip at the clock start / stop), `yardline_resid` (median distance of the painted yard lines from our 5-yd marks) |
| QC metrics | `clock_resid_p95`, `clock_coverage`, `bib_agree`, `panel_min_corr`, `max_cut`, `turf_min`, `turf_median`, `lane_frac`, `track_frac`, `start_support`, `max_step_yd` |

`final_time` and `split10` are the broadcast's unofficial times. They are
metadata and checks, not features.

## Inspecting a run

The measurement can be drawn on the video for any run. There are two views: a
contact sheet of six frames, and a video.

```bash
uv run python src/overlay.py <run_id>                    # contact sheet -> frames/qc/<run_id>.jpg
uv run python src/overlay.py --video <run_id>            # annotated clip -> frames/qc/<run_id>.mp4
uv run python src/overlay.py --video --speed 1 <run_id>  # real time (default is half speed)
uv run python src/overlay.py --video --pad 0.5 <run_id>  # 0.5 s of context before the start and after the stop
```

The video shows the broadcast frame with the measurement drawn on it:

- **red / blue circles:** the black dashes of the near / far lane row, numbered with
  their lane index;
- **yellow lines:** every 5 yd across the lane, placed from the fitted lane geometry;
- **green skeleton:** the tracked runner;
- **magenta dot and line:** the measured point, the ground straight below the hip;
- **top-left label:** clock time and the raw per-frame position, before smoothing.

The video covers the same window as the dataset: the clock window plus one frame
either side. With `--pad`, the extra context frames are marked "outside dataset
window".

The right-hand panel shows that frame's row of the dataset, with curves filling in
as the run goes. It shows the run id, athlete, class and split, the frame number and
`t_clock`, every channel's value, and running plots of position, velocity, trunk
angle, hip height and lead-knee angle. To render any other run:
`uv run python src/overlay.py --video <run_id>`.

A GIF for slides can be made from the MP4:

```bash
ffmpeg -i frames/qc/<run_id>.mp4 -vf "fps=12,scale=920:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse" frames/qc/<run_id>.gif
```

`frames/` is gitignored; renders stay local.

## Pipeline

```bash
uv sync
uv run python src/download.py      # 1. videos at 1080p -> video/ (gitignored)
uv run python src/segment.py       # 2. clock scan, runs, frame-accurate clock -> data/runs_raw.csv
uv run python src/positions.py     # 3. panel position code per run -> data/panel_positions.csv
uv run python src/extract.py       # 4. pose + lane on every frame -> cache/frames/<run_id>.pkl
uv run python src/build_db.py      # 5. tracking, yardage, channels, QC, split -> data/
uv run python src/calibrate.py     # 6. check the dash period against the painted yard lines
uv run python src/baseline.py      # 7. 1-NN sanity check per channel group
```

| Step | Time (Apple silicon) | Notes |
|---|---|---|
| `download.py` | network-bound (~3.5 GB) | retries alternate YouTube clients when one returns 403 |
| `segment.py` | ~1 min per video | `--videos=id1,id2` to limit; ids starting with `-` need the `=` form |
| `positions.py` | ~1.5 min | `harvest` / `label` subcommands rebuild the templates |
| `extract.py` | ~25 s per run, ~2 h for 303 | resumable; run two workers on disjoint `--videos` to halve it; `--lane-only --workers 6` recomputes lanes and yard lines on the cache without rerunning pose (~20 min) |
| `build_db.py` | ~10 s | everything downstream of the cache, so method changes rerun in seconds |
| `calibrate.py` | ~30 s | fits yards per dash period from every yard-line crossing, overall and per video |

The digit and position templates are committed. To rebuild them, run
`panel.py harvest` or `positions.py harvest`, which writes a sheet of cluster
prototypes. Label it once as a JSON list, then run `label`. Pose weights
(`yolo11m-pose.pt`) are downloaded by Ultralytics into `models/` (gitignored).

## How it is measured

### Time: the broadcast clock

The lower-third graphic carries a live 40-yd clock with 0.01 s resolution.

- **Reading.** Digits are read by template matching against glyph prototypes
  harvested from the feed and labelled once (`panel.py`). A field is accepted only
  as `d.dd` with its decimal point.
- **Finding runs.** A 4 fps scan of the whole video finds every stretch where one
  field counts up at real-time rate and then freezes on a plausible 40 time
  (`segment.py`).
- **Frame-accurate zero.** Every frame of the run is then read, and the clock zero
  `k0` is fitted to a fraction of a frame, with `t = (frame − k0) / 29.97`. The
  slope is fixed by the frame rate, so only the origin is fitted, and the median is
  immune to the odd misread. Residual: ~0.012 s p95, which is the graphic's own
  render jitter.
- **Layouts.** The layout differs by group: `40-YD DASH`, `1ST RUN / 2ND RUN`, or
  `10-YD SPLIT / 40-YD DASH`. The live field is simply the one that changes. The
  10-yd split does not count live: its slot holds 0.00, then shows the split.

The decoded clip for each run spans from 1 s before the clock start to 0.6 s after
the stop, so that smoothing and the start-line search have frames on both sides. The
dataset keeps the clock window plus one frame either side.

### Identity

- **Bib.** The bib number after the position code is read with the same glyph
  method and majority-voted over the run.
- **Position code.** The code itself (`WR`, `DB`, `QB`, …) is matched against
  labelled patches (`positions.py`). A run whose code isn't the session's group is
  excluded.
- **Athlete key.** `<code>-<bib>`. Bibs are unique within a position group.
- **Duplicates.** The same athlete and time appearing in a second upload would be a
  seam duplicate. None occurred. Two attempts by one athlete with identical times do
  happen and are kept: their footage differs.

### Position: the black lane marks

The lane is a white strip with two staggered rows of black dashes at a constant
spacing, one period. The period is 1.8 m (1.9685 yd; see Checks). Position is
measured against these marks on every frame (`lane.py`, `yardage.py`):

1. **Dashes and yard lines.** Dark elongated blobs surrounded by white lane are the
   dashes. Their shared angle is the lane direction. Transverse dark marks are kept
   as start-line candidates. The field's painted yard lines are found too: long
   white Hough lines on the turf, crossing the lane direction.
2. **Rows.** Two RANSAC lines split the dashes into the near and far row. Pieces of
   one dash cut by a leg or tripod are merged.
3. **Per-frame lattice.** Each row's dashes are indexed along the row, and an exact
   1-D perspective map `u = (a·s + b) / (c·s + 1)` from index to image position is
   fitted. Equally spaced points on a line obey this map exactly under perspective.
   Dashes that sit off the lattice, such as partly occluded ones, are dropped.
4. **Same index across frames.** Per-frame indices have an arbitrary offset. Offsets
   are chained on the rule that the camera moves less than half a period per frame.
   A second pass then removes any slip on the far row: the runner cannot move a
   whole period (~2 yd) in one frame.
5. **Across the lane.** The runner runs along the far row, so that row's map carries
   the measurement. A rung is the image of a line across the lane.
   - **Direction:** the painted yard lines are exactly that. Where they meet the far
     row, their angle is fitted linearly along the row and read at the runner, then
     median- and Savitzky–Golay-smoothed over time. They are found on ~96% of
     frames.
   - **Length:** the rung ends on the near row's line.
   - An earlier version paired the two dash rows to get the direction. That was off
     by 13–30°, because a small error in the rows' stagger swings it.
6. **Zero.** The start line is the cross-lane mark seen at the runner's hands at
   the clock start, measured where it crosses the far row. It coincides with a
   painted yard line (the goal line).
7. **Measured point.** The ground point straight below the hip centre. It is solved
   as the lane position whose rung, at the runner's lateral line, lies under the
   hip in the image. The runner's lateral line is the median across-lane position
   of the ankles over the run. This point follows the body's centre rather than the
   swinging feet, and lies on the ground plane.
8. **Cleaning.** Single-frame outliers more than 0.4 yd off a rolling median are
   removed, and gaps of up to 4 frames are filled.

The result is independent in every frame: nothing is integrated over time, so
errors do not accumulate. The clock is never used to scale distance.

**Scale.** The dash period is the lane's own unit. Its length in yards comes from
the painted yard lines, which are 5 yd apart, with the start line on one of them.
`calibrate.py` places every yard line that crosses the far row in lane periods and
fits the scale. Over 50k crossings it gives 1.9679 yd (1.7995 m) per period; the
per-video fits run from 1.9655 to 1.9698. That is a 1.8 m layout, used exactly:
`YD_PER_PERIOD = 1.8 / 0.9144` in `features.py`. An earlier assumption of 2.0 yd
put the 40-yd mark about 0.9 yd short at the finish.

### Pose and tracking

- **Pose model.** YOLO11m-pose runs at 1280 px on every frame, on the Apple GPU (MPS).
- **Seeding.** The runner is the person set on the lane at the clock start: ankles
  and wrists between the dash rows.
- **Following.** The runner is followed by continuity of the hip and shoulder
  centres, which survive truncated boxes and don't swing like limbs. The scale is
  normalised by the larger of torso length and box size, because the torso is
  foreshortened in the stance. Candidates whose legs are off the lane get a penalty;
  this is what keeps seated officials at the timing tables from being picked up.
- **Smoothing.** Keypoints below 0.3 confidence are masked, gaps of up to 4 frames
  are filled, and each coordinate is Savitzky–Golay smoothed.

### Channels

- **Kinematics.** `x_yd` is smoothed with a 9-frame window. `v_yds` and `a_yds2` are
  15- and 21-frame Savitzky–Golay derivatives.
- **Angles.** Angles are measured in the image plane. "Forward" is the image
  direction of the lane at the runner, and "vertical" is image-up.
- **Hip height.** Measured to the ground point under the hip, divided by the
  run-median leg length (thigh + shank).
- **Foot contact.** Uses the raw ankle keypoints, because a contact lasts only ~3
  frames and smoothing would erase it. An ankle is planted when its speed along the
  lane is below 35% of body speed (minimum 1.5 yd/s).
- **Steps.** A step is a touchdown at a new place along the lane, at least
  0.12 yd × body speed ahead of the previous one. That rule survives left/right
  swaps of the keypoints.

## Quality control

A run is kept only if every rule holds (`src/qc.py`); `qc_reason` lists every failed
rule. The window checked is the dataset window: the clock window plus one frame
either side. A transition that ends before that window does not reject a run.

| Check | Rule |
|---|---|
| Standard broadcast view | panel logo correlation ≥ 0.85 on every frame; no shot cut (colour histogram Bhattacharyya ≤ 0.35); turf covers ≥ 18% of every frame and ≥ 40% at the median |
| Clock | fit residual p95 ≤ 0.025 s; clock read on ≥ 60% of run frames; bib agreement ≥ 80% |
| Measurement | lane on ≥ 90% and runner on ≥ 90% of run frames; start line seen on ≥ 5 frames; no single-frame jump over 0.8 yd |
| Consistency | hip within 1.5 yd of 40 at the clock stop; hip between −2.2 and 0.1 yd at the clock start |
| Reported only | `yardline_resid`: median distance of the painted yard lines from our 5-yd marks |
| Identity | panel position code matches the session group |

The 47 rejections break down as:

| Cause | Runs |
|---|---|
| Non-standard view | 27 |
| Clock stop disagrees with the measured distance | 17 |
| Lane or runner lost | 1 |
| Hip out of range at the start | 1 |
| Clock residual | 1 |

- **Non-standard view.** Split screens with the commentators (for example the end of
  the EDGE session) and graphic transitions around the start.
- **Clock disagreements.** For example a 3.94 s reading with the runner at 33 yd.
  The distance check doubles as a clock-error detector.

## Checks

- **Painted yard lines.** They are the calibration target, and also a per-run
  check (`yardline_resid`): a median 0.16 yd per run (p90 0.20).
  - The residual is flat along the run: medians at the 15, 25, 30 and 35 lines lie
    within ±0.1 yd.
  - Lines at 10/20/30/40 yd share their spot with the painted distance mats and
    field numbers, whose white edges add false detections. That inflates the spread,
    not the medians.
- **Stop position.** The clock is not used for scale. At the clock stop the hip is
  at 39.60 ± 0.24 yd over all kept runs. Every position lands between 39.48 and
  39.71, with sd 0.12–0.17 (OL 0.39). This is expected: the timer stops when any
  part of the athlete breaks the beam, and the leading body parts are a few tenths
  of a yard ahead of the hip.
- **Held-out 10-yd split.** The panel split is never an input. Over 39 lineman runs
  the hip crosses 10 yd +0.128 ± 0.024 s after it. Leaning ~45° at that point, the
  head and shoulders break the beam well ahead of the hip.
- **Physics, from class medians on the distance grid.**
  - Top speed: SKILL 11.7 yd/s (10.7 m/s), STRONG 11.3, LINEMAN 10.1.
  - Trunk lean falls from ~40–44° at 1 yd to 10–13° at 30 yd in every class.
    Linemen are the most upright.
  - `hip_height_ratio` rises from ~0.63 to ~1.02, nearly the same in every class.
  - Step length from 30 to 39 yd: 2.4 yd (LINEMAN) to 2.5 yd (SKILL, STRONG).
- **Visual.** The contact sheets and videos above, spot-checked on random kept runs
  and on every rejection category.
- **Signal.** 1-NN on the time grid, train → test, per channel group
  (`baseline.py`, or `--grid distance`; 3 classes, chance ≈ 0.33):

| Channels | Accuracy |
|---|---|
| progress (`t_clock`, `x_yd`) | 0.74 |
| velocity + acceleration | 0.70 |
| trunk angle + hip height | 0.59 |
| joint angles | 0.40 |
| stride (frequency, length) | 0.56 |
| all pose channels | 0.43 |

Pose separates the classes less than speed does, but above chance. Plain 1-NN
over all pose channels is weak and moves by several points between builds. This is
a sanity check, not a model.

## Limitations

- **Pose angles are image-plane angles.** The camera is oblique and moving, so they
  are not true sagittal-plane joint angles. They are comparable across runs because
  the broadcast camera follows the same path every time, but the camera's
  viewpoint changes along the run.
- **Contacts and steps are coarse.** At 29.97 fps a ground contact spans ~3 frames.
  `step_freq_hz` is quantised to 30/n Hz (3.75, 4.29, 5.0, …).
- **`x_yd` is a mark count.** It is lane periods × 1.9685; the dash marks are what
  is measured. Its zero is the start line, so at the clock start the hip is at
  −1.21 ± 0.16 yd. That start value depends on the runner's lateral line, which the
  oblique view at the start makes sensitive.
- **The clock is unofficial.** It is the broadcast's "unofficial times" graphic,
  used as the time base, not the official result.

## Repository layout

| Path | Contents |
|---|---|
| `src/download.py` | fetch videos |
| `src/panel.py` | panel band decoding, glyph harvesting and labelling, clock and bib reading, position-code patches |
| `src/segment.py` | run detection and frame-accurate clock fit |
| `src/positions.py` | position code per run |
| `src/extract.py` | per-frame pose detections, lane geometry and QC signatures, cached |
| `src/lane.py` | dash and yard-line detection, row fitting and indexing, 1-D perspective map |
| `src/yardage.py` | runner tracking, cross-frame indexing, rung from yard lines, start line, hip position |
| `src/calibrate.py` | yards per dash period from the painted yard lines |
| `src/features.py` | per-frame channels |
| `src/qc.py` | QC metrics and rules |
| `src/build_db.py` | dataset assembly, duplicates, split, clock-window trim, time and distance grids |
| `src/overlay.py` | contact sheets and annotated videos |
| `src/baseline.py` | 1-NN sanity check |
| `video/`, `cache/`, `frames/`, `models/` | gitignored: footage, per-run caches, renders, weights |

## Data & licensing

No video, frames or renders are committed; footage is NFL-copyrighted. The repo ships
video IDs, small digit/position templates and derived numeric measurements only.
Code MIT; derived measurements CC BY 4.0.
