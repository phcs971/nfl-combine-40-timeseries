# NFL Combine 40-Yard Dash — Position-Time Series

Building a dataset of **position vs. time** for NFL Combine 40-yard dashes, sampled to
represent three athlete classes rather than the fastest athletes in each.

Position series are differentiated into velocity and acceleration via a fitted
sprint model, enabling time-series analysis across body types.

## Athlete classes

| Class | Positions | Frame n | Drafted | Undrafted | 40 mean ± sd |
|---|---|---|---|---|---|
| `skill` | WR, CB, SAF | 146 | 109 | 37 | 4.449 ± 0.086 |
| `strong` | RB, TE, LB, EDGE | 132 | 99 | 33 | 4.606 ± 0.134 |
| `line` | OT, G, C, DT | 102 | 79 | 23 | 5.088 ± 0.149 |

QBs and specialists (K, P, LS) are excluded. Seasons 2025 + 2026; a single year
does not survive video attrition.

Target: **20 athletes per class**, stratified by 40-time tercile × draft status.

## Design decisions

**The 2025 "FULL" videos are the primary corpus.** The NFL channel publishes
position-group sessions, not per-player clips. The 2025 uploads are titled `FULL` and
run 77-108 seconds per athlete, uniformly across every position group; the 2026
uploads run 35-57 s/athlete. That uniformity is the signal — a curated highlight reel
would vary with position popularity, an uncut session would not. The 2025 videos
therefore contain every athlete who ran, including the slow and the undrafted, which
is what makes representative sampling possible. 2026 is treated as a trimmed
supplement, not an equal source.

**Draft status is rebuilt, never read directly.** The nflverse combine release ships
`draft_round` unpopulated for the most recent class — all 319 of the 2026 rows are
null. Taken at face value that reads as "everyone went undrafted." Status is joined
from the `draft_picks` release on `pfr_id`, with a name-normalized fallback for the
~8% of rows carrying no `pfr_id`.

**Sub-frame interpolation is required, not optional.** The entire corpus is 30 fps
only - no 60 fps rendition is offered for any of the 20 videos. Snapping a yard-line
crossing to the nearest frame quantises it to 33 ms, and at top speed an athlete
covers ~0.33 m in that interval. Crossing times are therefore recovered by
interpolating the landmark's pixel position across the two bracketing frames, which
puts timing resolution well below the frame interval.

**Splits are fitted, not finite-differenced.** Nine yard-line crossings over ~4.5 s
differentiated twice yields acceleration dominated by timing noise. Instead a
mono-exponential velocity model is fitted to the split times:

```
v(t) = v_max · (1 − e^(−t/τ))
x(t) = v_max · (t + τ·e^(−t/τ) − τ)
```

Two parameters over ~9 points give smooth analytic `v(t)` and `a(t)`, plus
interpretable outputs (`v_max`, `τ`, peak horizontal force, `P_max`). This is the
standard Samozino/Morin sprint-profiling approach.

## Video corpus

20 NFL-channel position-group videos, 6.97 h total, all 30 fps
(`data/videos.csv`, validated by `src/check_videos.py`).

| Class | 2025 `FULL` groups | 2026 trimmed groups | Frame n (2025) |
|---|---|---|---|
| `skill` | WR, CB, SAF | WR x2, DB, SAF | 78 |
| `strong` | RB, TE, LB, DE | RB, TE, LB, EDGE | 72 |
| `line` | OL, DL | OL x2, DL | 48 |

Each class draws on multiple position groups, so no class is a single-position
artefact. 2025 alone clears the 20-per-class target in every class.

## Open problem: segmentation and identification

The videos carry no chapters and a boilerplate description, so run boundaries are not
free. Each athlete must be located within a 20-53 minute session and matched to a
frame row. The intended route is OCR of the broadcast lower-third graphic (name,
school, official time) on sampled frames, building a who-is-on-screen timeline, then
isolating the live run from its replays within each athlete's segment.

The per-athlete durations suggest both of an athlete's two combine attempts may be
shown. If so that yields two independent measurements per athlete and a
test-retest reliability estimate - to be confirmed during segmentation.

## Known error sources

- **Slow-motion replays.** Combine broadcasts cut to slo-mo constantly, and a replay
  is frame-indistinguishable from a live run. Every clip needs explicit real-time
  verification.
- **Frame rate.** Confirmed 30 fps for all 20 videos; no 60 fps rendition exists.
  This is the hard floor on timing precision, mitigated by sub-frame interpolation.
- **Parallax.** The dominant error, ahead of frame rate. Runners are offset from the
  camera axis, so "crosses the yard line" carries an offset that varies down the
  track. A fixed landmark (hip center) is used and the offset treated as estimable
  bias.
- **Timing convention.** Official combine 40s use a human-triggered start and laser
  finish. Video-derived times start at first movement and will not match official
  times exactly — this is expected, and used as a validation signal.

## Data & licensing

No video or extracted frames are committed. Broadcast footage is NFL-copyrighted and
downloading it conflicts with the host platform's terms. This repo ships **URLs,
timestamps, and derived numeric measurements only**, with scripts to reproduce
locally — the same approach used by video datasets such as Kinetics.

Combine and draft data come from [nflverse](https://github.com/nflverse/nflverse-data).

## Usage

```bash
uv run python src/build_frame.py --seasons 2025 2026   # -> data/frame.csv (380 athletes)
uv run python src/check_videos.py                      # validate the 20-video registry
```

## Status

- [x] 1 — Sampling frame from combine + draft data
- [x] 2 — Video inventory: 20 videos, 6.97 h, all validated
- [ ] 3 — Stratified sample, 20/class, validated against population
- [ ] 4 — Frame extraction, fps + real-time verification
- [ ] 5 — Yard-line crossing annotation
- [ ] 6 — Sprint-model fit → position/velocity/acceleration series
- [ ] 7 — Validation: derived vs. official 40 times

Code is MIT licensed. Derived measurements are released under CC BY 4.0.
