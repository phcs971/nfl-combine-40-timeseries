# NFL Combine 40-Yard Dash — Position-Time Series

Position vs. time series for NFL Combine 40-yard dashes, sampled to represent three
athlete classes rather than the fastest athletes in each. Series are differentiated
into velocity and acceleration via a fitted sprint model.

Scope: **2026 combine**.

## Athlete classes

| Class | Positions | Frame n | Drafted / undrafted | 40 mean ± sd |
|---|---|---|---|---|
| `skill` | WR, CB, SAF | 68 | 52 / 16 | 4.439 ± 0.084 |
| `strong` | RB, TE, LB, EDGE | 60 | 47 / 13 | 4.602 ± 0.133 |
| `line` | OT, G, C, DT | 54 | 44 / 10 | 5.071 ± 0.147 |

QBs and specialists excluded. Target: **20 athletes per class**, stratified by
40-time tercile × draft status. Undrafted athletes are the binding constraint,
`line` most of all at 10.

## Video corpus

11 NFL-channel position-group videos (`data/videos.csv`), all 30 fps. Each class
draws on multiple position groups, so no class is a single-position artefact.

Videos were uploaded within days of the combine, ~7 weeks before the draft, so video
inclusion is independent of draft outcome. Selection on 40 time is not ruled out by
that and is checked during segmentation by comparing athletes found on video against
the frame.

## Athlete identification

Bibs are assigned alphabetically by last name within a position group, over every
invitee in that group including those who never post a 40. Reading the two-digit bib
and indexing that roster is cheaper and steadier than OCRing the name, which would
need a full alphabet of prototypes. Confirmed against five bibs spanning the
alphabet; suffixes must be stripped before sorting or `Keenan III` and `Jackson Jr.`
sort under their suffix and shift every bib after them.

The bib renders smaller than the timing digits and correlates lower against the same
prototypes, so it is majority-voted across frames of a run rather than trusted from
any single frame.

## Run manifest

`data/runs.csv` — 328 runs across the 11 videos, every run bound to an athlete
(327/328 with a unanimous bib vote). 180 are marked `complete`; the rest are
fragments where clock tracking was lost mid-run, whose final value is not a 40 time.
Two runs are flagged `duplicate`: consecutive group uploads overlap at their seam,
the later video opening by replaying the athlete the previous one closed on.

Filtering to complete, non-duplicate runs gives **180 runs over 134 athletes**:

| Class | Runs | Athletes |
|---|---|---|
| `skill` | 76 | 57 |
| `strong` | 54 | 41 |
| `line` | 50 | 36 |

Over the 130 athletes with an official time, the best of their attempts sits within
0.05 s of it for 119, mean +0.016, sd 0.047.

Bib offsets are fitted per video rather than hard-coded. Two constraints pin them:
every observed bib must land inside the roster, and it must land on someone who
actually posted a 40 — an athlete with no recorded time cannot be the one on screen.
Timing alone does not separate offsets when a group's times cluster tightly. The
fitted offsets recover the real block structure: safeties sit 32 behind the
cornerbacks they share the `DB` group with, and edge rushers 29 behind the defensive
tackles in `DL`.

## Position tracking

The camera pans with the athlete, so image position alone measures nothing. The
lane's black dashes are fixed in the world, so their frame-to-frame shift measures
the pan and the runner's displacement is the remainder:

    displacement = (shift of foot in image) - (shift of marks in image)

projected onto the lane axis, with scale from local mark spacing so perspective
cancels. Distance accumulates in mark-spacings, and the yard value of one spacing
follows from the run being 40 yards — no external calibration.

The dashes sit in two rows offset along the lane. Pooling them and taking the median
gap mixes true spacings with the offset between rows, and the mixture shifts as rows
enter and leave frame, which distorts the recovered speed profile rather than merely
rescaling it. Spacing is therefore measured within a single row.

### Detectors

`src/detectors.py` keeps the two measurements apart, since they want opposite image
treatment: the yard ticks are small static marks found by colour-thresholding the
field, the athlete is a large deformable object needing a learned pose model.

`detect_athlete` runs YOLO pose and takes the lower ankle as ground contact - it
shares the ground plane with the ticks, so the projection carries no parallax term.
Candidates are filtered to the middle of frame, where the tracking camera holds the
runner, and to feet low enough to be on the lane; without the second test the model
picks officials standing further up the field. It resolves the runner on 15/15
frames of the reference run.

`detect_yard_ticks` fits a RANSAC line to white marks on the field. The field
carries several parallel rows - the sideline ticks and two inbound hash rows - and
picking whichever has most inliers makes the fit jump between rows, taking the
apparent spacing with it. Passing the runner's foot selects the row he runs beside,
which turns spacing from a 84-172 px jitter into a smooth 126-172 px perspective
trend.

`detect_mats` finds the numbered distance mats by their saturated yellow border
around a black face. The border alone also matches the lane's painted yellow lines,
so a mat is required to be dark inside — that separates them cleanly (dark fraction
~0.2-0.4 against ~0.0). A mat's numerals split its face, so it arrives as two blobs
a few hundred pixels apart and they must be merged; ten yards is an order of
magnitude wider, so anything closer is the same mat.

### Crossings and the mat offset

`src/crossings.py` times when the foot passes each mat. A crossing is a coincidence
of runner and mat within one frame, so it is immune to the camera panning, and both
lie on the ground plane so it carries no parallax term.

**The numbered mats are not at the dash's 10/20/30 marks.** At his true 10-yard
moment the reference athlete is still well short of the `10` mat and reaches it 0.65 s
later. The mats are 10 yards apart, but the first sits about 13 yards from the
start, so their distance from the start is carried as one unknown per video.

That offset is one physical constant for a video, which makes agreement between
independently fitted runs a check on the geometry: the two runs that saw all four
mats fit it at **13.03 yd with sd 0.009**, and returned near-identical `v_max` and
`tau`. Holding it fixed then lets runs that saw fewer mats be fitted, provided the
index of the first mat seen is inferred rather than assumed — a run that misses the
first mat otherwise has every crossing assigned ten yards short.

Mats are tracked across frames and each track's own sign change is interpolated.
Comparing consecutive frames instead loses a crossing whenever mat detection blinks
out at the moment of passing, which is exactly when the runner occludes the mat.

## Results

`data/fits.csv` — one row per run; `data/series.csv` — position sampled every 50 ms
for the runs that fit.

**148 of 180 runs fit cleanly, covering 111 athletes**, every class well above the
20-athlete target. Residuals: median 0.12 yd, p90 0.31 yd.

| Class | Runs | Athletes | `v_max` (m/s) | `tau` (s) |
|---|---|---|---|---|
| `skill` | 58 | 43 | 10.65 ± 0.55 | 1.02 |
| `strong` | 45 | 36 | 10.50 ± 0.63 | 1.15 |
| `line` | 45 | 32 | 9.06 ± 0.43 | 1.06 |

Crossings are not necessarily consecutive mats — one can be missed when the runner
occludes it, and a stray detection can add one that is not a mat. Fitting over which
mats the crossings correspond to, rather than forcing them to be consecutive, is what
took this from 102 runs to 148; the failures had been sitting in a tight band near
half a mat spacing rather than scattering, which is the signature of a mis-assigned
index rather than noise.

Three things say the geometry is right:

- **The mat offset agrees across all 11 videos** — fitted independently per video
  from different athletes, it lands at 12.94 yd, sd 0.25, range 12.46-13.36.
- **Class ordering and magnitude** come out as physics requires, skill fastest and
  line slowest, without either being an input.
- **Held-out 10-yd split.** The panel publishes a 10-yd split for `line` groups and
  it is never used in the fit. Over 21 runs the model reaches 10 yd **+0.109 s**
  later than the panel, sd **0.029**. The scatter is small and every run has the
  same sign, so this is a calibration constant rather than noise — most likely the
  mono-exponential understating the drive phase out of a three-point stance, which
  is a known weakness of the model near t=0. Corrected for it, 18 of 21 fall within
  0.05 s.

The 32 runs that remain unfitted see one mat or none. Detection is not the limit:
lane and athlete resolve on 100% of frames and mats on ~75%.

A step-by-step visual walkthrough of the pipeline is published as an artifact,
generated by `src/make_filmstrips.py`.

## Method

**Draft status is joined from the `draft_picks` release, not read from the combine
release** — the latter ships `draft_round` null for all 2026 rows, which reads as
universally undrafted. A name-normalised fallback covers rows with no `pfr_id`.

**The broadcast clock is the time reference.** The graphic carries a live 40-yd clock
at 0.01 s resolution that starts on the athlete's first movement and freezes at the
finish. It gives `t=0` and a continuous check on elapsed time without inferring
either from pixels, and it drives segmentation: clock at `0.00` is pre-run, counting
is the live run, frozen is post-run. Panel layout is position-dependent — `line`
groups show `10-YD SPLIT | 40-YD DASH`, other groups show `40-YD DASH` alone, at a
different x offset — so the read region is located per group, not by a fixed crop.
Digits are large, fixed-font and high-contrast, so template matching is used rather
than general OCR — prototypes in `data/glyph_templates.npz`, several per digit to
absorb the feed's subpixel anti-aliasing. Reading is self-validating: the clock must
advance by exactly one frame interval per frame, so a misread digit shows up as a
backwards or oversized step rather than passing silently.

Displayed times are labelled `UNOFFICIAL` and run 0.00-0.07 s off the official times
in the frame; the frame is authoritative.

**Sub-frame interpolation.** The corpus is 29.97 fps (30000/1001) with no 60 fps
rendition. Nearest-frame snapping quantises a crossing to 33.4 ms, ~0.33 m at top
speed. Crossing times are recovered by interpolating the landmark's pixel position
across the two bracketing frames.

**Per-frame calibration against a panning camera.** The live camera tracks the runner
rather than holding a fixed view, so no single homography covers a run. A yard-line
crossing is nonetheless camera-motion-invariant: it is a coincidence of runner and
line within one frame. Scale is recovered per frame from the yard markings visible in
that frame, whose 5-yd spacing is known.

**The tracked landmark is ground contact, not hip.** Feet share the ground plane with
the yard lines, so their coincidence in image space is a coincidence in reality and
carries no parallax term. A hip or torso landmark sits ~1 m above the plane and
projects ahead of or behind the true position under an elevated camera. Stride
oscillation in the foot signal is absorbed by the model fit.

**Splits are fitted, not finite-differenced.** Nine yard-line crossings over ~4.5 s
differentiated twice yields acceleration dominated by noise. A mono-exponential
velocity model is fitted to the split times:

```
v(t) = v_max · (1 − e^(−t/τ))
x(t) = v_max · (t + τ·e^(−t/τ) − τ)
```

Two parameters over ~9 points give smooth analytic `v(t)` and `a(t)`, plus `v_max`,
`τ`, peak horizontal force and `P_max` (Samozino/Morin sprint profiling).

## Coverage

Verified on the 2026 DL session: all 15 athletes in that group appear in the video,
including the slowest (5.31) and all three undrafted. No evidence of selection on
speed or draft status. Athlete name and bib are on screen continuously, so the
who-is-on-screen timeline is readable at any sample rate.

Most athletes run twice: the DL session yields 28 runs across its 15 athletes, 13 of
them with a second attempt. The panel shows `1st ATT` / `2nd ATT`, with `---` where
there is no second.

Segmentation and identification are verified end-to-end on that session: 15/15
athletes recovered with unanimous bib agreement, and the best of each athlete's
attempts sits within 0.03 s of their official time (mean +0.005, sd 0.010).

## Known error sources

- **Replays and cutaways.** The feed cuts to other angles and to slow motion within
  seconds of the finish. The running clock discriminates these from the live run;
  frozen or absent clock means the frame is not measurable.
- **Stride oscillation** in the ground-contact landmark, absorbed by the model fit
  rather than differentiated directly.
- **Timing convention.** Official 40s use a human-triggered start and laser finish;
  video-derived times start at first movement and will not match exactly. Used as a
  validation signal.

## Data & licensing

No video or extracted frames are committed — footage is NFL-copyrighted. This repo
ships URLs, timestamps and derived numeric measurements only, with scripts to
reproduce locally. Combine and draft data from
[nflverse](https://github.com/nflverse/nflverse-data).

## Usage

```bash
uv run python src/build_frame.py --seasons 2026   # -> data/frame.csv
uv run python src/check_videos.py                 # validate video registry
uv run python src/probe_video.py video/<id>.mp4   # contact sheet for inspection
uv run python src/read_clock.py video/<id>.mp4 --ss 2 --dur 6   # read timing panel
uv run python src/build_runs.py                   # -> data/runs.csv (resumable)
uv run python src/build_series.py                 # -> data/fits.csv, data/series.csv
```

`probe_video.py` needs a local download; `video/` and `frames/` are gitignored.

Code MIT. Derived measurements CC BY 4.0.

## Status

- [x] Sampling frame from combine + draft data
- [x] Video registry
- [x] Feed structure verified (clock, coverage, camera behaviour)
- [x] Clock reader (`src/read_clock.py`), validated against the frame interval
- [x] Athlete identification from bib (`src/read_bib.py`)
- [x] Run manifest for all 11 videos (`src/build_runs.py` -> `data/runs.csv`)
- [x] Mat crossings and sprint-model fit (`src/build_series.py`)
- [ ] Recover the 32 runs that see one mat or none
- [ ] Stratified sample, 20/class, validated against the population
- [ ] Stratified sample, 20/class
- [x] Mat crossings and sprint-model fit (`src/build_series.py`)
- [ ] Recover the 32 runs that see one mat or none
- [ ] Stratified sample, 20/class, validated against the population
- [ ] Sprint-model fit
- [ ] Validation vs. official 40 times
