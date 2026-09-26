# Datasheet: NFL Combine 40-Yard Dash Movement Time Series (2026), v1.0.0

Written following Gebru et al., *Datasheets for Datasets* (Communications of the ACM,
2021). The method is described in full in the [README](README.md); the column-level
data dictionary is [`datapackage.json`](datapackage.json).

## Motivation

**Why was the dataset created?** To study whether an athlete's position class can be
recognised from how they sprint. It was created as a university project on
time-series classification. Each 40-yard dash from the 2026 NFL Scouting Combine
broadcasts becomes a multivariate time series, labelled with one of three classes:
- SKILL: WR, CB, SAF;
- STRONG: RB, TE, LB, EDGE;
- LINEMAN: OL, DT.

**Who created it?** Pedro Henrique Cordeiro Soares
([ORCID 0009-0000-6041-2494](https://orcid.org/0009-0000-6041-2494), phcs.971@gmail.com).

## Composition

**What are the instances?** One instance is one 40-yard dash attempt (a "run") by one
athlete.

**How many are there?**
- 326 runs were detected in 11 videos: 276 kept, 48 rejected by quality control,
  2 excluded as a QB.
- The 276 kept runs cover 164 athletes: SKILL 61, STRONG 54, LINEMAN 49.
- By position: WR 50, CB 38, SAF 22, RB 15, TE 32, LB 18, EDGE 24, OL 56, DT 21.

**What does each instance contain?**
- A per-frame series at 29.97 Hz, 130–164 frames long. It runs from one frame before
  the on-screen clock starts to one frame after it stops.
- Channels:
  - position along the lane (`x_yd`), velocity and acceleration;
  - dimensionless pose channels: trunk angle, hip height ratio, knee, hip and elbow
    angles, shin angle, thigh separation and arm swing;
  - foot contact, step frequency and step length.
- Two resampled views of the same series: 101 points from clock start to stop, and
  every 0.25 yd from 0 to 39 yd.
- Run metadata:
  - identity: session position, panel position code, athlete key, attempt;
  - the class label and the train/test split;
  - the unofficial broadcast 40-yd time (`time_40yd`) and, for line groups, the 10-yd
    split (`time_10yd`);
  - quality-control metrics.

**Is it a sample?** It covers every run the NFL broadcast in these 11 YouTube uploads
of the 2026 combine 40-yard dash sessions, for the positions above. It is not every
combine participant:
- athletes who skipped the drill or whose attempts weren't shown are absent;
- QBs and specialists are out of scope.

What was shown is the broadcaster's choice, not ours.

**Labels.**
- The class comes from the position-group session the video covers.
- The position code shown on the broadcast panel is read on every run. A run whose
  code isn't the session's group is excluded; this removed a QB who ran in the WR
  group 2 session.

**Missing values.**
- Pose and position channels are empty on frames where the runner or the lane marks
  weren't measured: `valid` = 0 on 3.5% of frames.
- Stride channels are empty before the first stride and after the last touchdown.
- `time_10yd` is present only where the panel shows a split: 56 of the 77 kept
  lineman runs.

**Relationships.**
- Attempts by the same athlete share the key `athlete` (`<panel code>-<bib>`).
- Runs link to their source video through `video_id`.

**Splits.** Train and test are split by athlete, so both attempts land on the same
side. The split is stratified by position within class and seeded (2026): at least 24
athletes per class on each side.

**Errors, noise and known biases.**
- **Position noise.** A painted field line that doesn't move measures with a
  per-frame standard deviation of 0.10 yd (p90 0.13) as the camera pans. The painted
  lines agree with our 5-yd marks to a median of 0.16 yd per run.
- **The measured point is the hip, not the body part that breaks the timing beam.**
  At the clock stop the hip is at 39.60 ± 0.23 yd.
- **The clock is the broadcast's "unofficial times" graphic.** Its fitted zero has a
  residual of ~0.012 s (render jitter). It isn't the official combine result.
- **Pose angles are image-plane angles** from an oblique camera that pans along the
  run. They are comparable between runs, but they aren't true sagittal-plane angles.
- **Foot contact is coarse.** At 29.97 fps a ground contact spans ~3 frames, so
  `step_freq_hz` takes only discrete values of 30/n Hz.
- **The classes reflect position groups, not individual roles.** Sample sizes per
  position are uneven (RB 15 to OL 56).

**External resources.** The raw footage is not included. The source videos are listed
in `data/videos.csv` with the title, channel, upload date, URL and the SHA-256 of the
exact file analysed. They may be edited or removed by YouTube or the NFL at any time.
The released measurements don't depend on them remaining available.

**Confidential or offensive content.** None.

**Does it identify people?** Indirectly.
- No names are stored.
- The athlete key (for example `DB-14`) together with the public combine roster
  identifies the athlete.
- The data describe public athletic performances that were broadcast publicly.
- Pose kinematics are recorded as movement measurements, not for identification.

## Collection process

**How was the data acquired?**
- The 11 public videos were downloaded from the NFL YouTube channel at 1080p with
  yt-dlp. They were uploaded between 2026-02-26 and 2026-03-01.
- All measurements are automatic, from the video pixels:
  - the clock and bib by template matching;
  - pose by Ultralytics YOLO11m-pose;
  - lane geometry by classical computer vision.
- No external timing, roster or tracking data were used.

**How was it validated?**
- The painted yard lines are the calibration target and a per-run check.
- Two further checks are independent of the position scale:
  - the hip position at the clock stop;
  - the panel's 10-yd split, which is never used as an input.
- Contact sheets and annotated videos were reviewed for random kept runs and for
  every rejection category.

**Who collected it, and when?** The author, with the code in this repository,
in September 2026. The combine itself took place in late February 2026.

**Consent and ethics.**
- Athletes were not contacted. The source is a public broadcast of a public event.
- No institutional ethics review is recorded for this derived-data release.
- A person who wants their runs removed can open an issue on the repository. Their
  runs will be excluded from the next release.

## Preprocessing, cleaning, labelling

The full pipeline is in `src/` and described step by step in the README.

**Cleaning.**
- Keypoints below 0.3 confidence are masked, and gaps of up to 4 frames are filled.
- Position outliers more than 0.4 yd from a 7-frame rolling median are removed.
- Channels are Savitzky–Golay smoothed.

**Quality control.** Runs are rejected for any of:
- non-standard broadcast views (split screens, transitions);
- clock anomalies;
- disagreement between the clock stop and the measured distance;
- lost lane or runner.

The rules are in `src/qc.py`; failed rules are recorded per run in `qc_reason`.

**Raw data.** The raw video is not distributed. Intermediate per-run tables
(`runs_raw.csv`, `panel_positions.csv`) are included.

**Reproducibility.**
- The code is in `src/`. Dependencies are pinned in `uv.lock`.
- The model weights SHA-256 is recorded below.
- The panel-reading templates are rebuilt deterministically from committed labels
  plus the recorded video files.

## Uses

**Intended.**
- Time-series classification research and teaching.
- Exploratory sprint biomechanics.
- Method development for measuring athletes in broadcast video.

**Not intended.**
- Scouting, grading, ranking or making decisions about individual athletes.
- Inferring health, injury or identity.
- Any use that redistributes the NFL footage.

**Things to know before use.**
- Classes are position groups, and classes differ strongly in speed. A classifier
  can reach high accuracy from time-to-distance alone (see the README baseline),
  so report which channels you use.

## Distribution

**Where.** The public GitHub repository
<https://github.com/phcs971/nfl-combine-40-timeseries>. Files can be loaded directly
from `raw.githubusercontent.com` URLs or opened in Google Colab (see the README).

**Licences.**
- Derived data: CC BY 4.0 (`data/LICENSE`).
- Code: MIT (`LICENSE`).
- Neither covers the NFL footage. Broadcast stills in `docs/img/` are reproduced at
  reduced resolution for research illustration, credited to the NFL.
- The pose model (Ultralytics YOLO11, AGPL-3.0) is a runtime dependency, not
  redistributed here.

**Third-party terms.** Downloading from YouTube is subject to YouTube's Terms of
Service. Anyone rerunning the pipeline is responsible for complying with them and
with local law.

## Maintenance

**Who maintains it?** The author, through the GitHub repository. Questions,
corrections and removal requests go through issues, or by email to
phcs.971@gmail.com.

**Updates.** Versions are recorded in [CHANGELOG.md](CHANGELOG.md). Releases are
tagged, and data values are fixed within a version.

**Errata.** Known issues are listed in the README (Limitations) and in the changelog.

## Provenance of tools and models

| Component | Version / identifier |
|---|---|
| Pose model | Ultralytics `yolo11m-pose.pt`, SHA-256 `29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2` |
| Python packages | pinned in `uv.lock` |
| Source videos | `data/videos.csv` (YouTube id, upload date, SHA-256 of the analysed file) |
| Lane scale | 1.8 m per dash period; `src/calibrate.py` fits 1.7996 m from 56,756 yard-line crossings (per video 1.9658–1.9700 yd) |
