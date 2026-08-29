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

## Method

**Draft status is joined from the `draft_picks` release, not read from the combine
release** — the latter ships `draft_round` null for all 2026 rows, which reads as
universally undrafted. A name-normalised fallback covers rows with no `pfr_id`.

**Sub-frame interpolation.** The corpus is 30 fps with no 60 fps rendition. Nearest-
frame snapping quantises a crossing to 33 ms, ~0.33 m at top speed. Crossing times
are recovered by interpolating the landmark's pixel position across the two
bracketing frames.

**Splits are fitted, not finite-differenced.** Nine yard-line crossings over ~4.5 s
differentiated twice yields acceleration dominated by noise. A mono-exponential
velocity model is fitted to the split times:

```
v(t) = v_max · (1 − e^(−t/τ))
x(t) = v_max · (t + τ·e^(−t/τ) − τ)
```

Two parameters over ~9 points give smooth analytic `v(t)` and `a(t)`, plus `v_max`,
`τ`, peak horizontal force and `P_max` (Samozino/Morin sprint profiling).

## Open problem: segmentation and identification

Videos carry no chapters and a boilerplate description. Each athlete must be located
within the session and matched to a frame row — intended route is OCR of the
broadcast lower-third (name, school, official time) on sampled frames, then isolating
the live run from its replays. Both combine attempts appear to be shown, giving two
measurements per athlete and a test-retest estimate.

## Known error sources

- **Slow-motion replays** are frame-indistinguishable from live runs; each clip needs
  explicit real-time verification.
- **Parallax** dominates over frame rate. Runners are offset from the camera axis, so
  a yard-line crossing carries an offset varying down the track. A fixed landmark
  (hip centre) is used and the offset treated as estimable bias.
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
```

Code MIT. Derived measurements CC BY 4.0.

## Status

- [x] Sampling frame from combine + draft data
- [x] Video registry
- [ ] Segmentation + player identification
- [ ] Stratified sample, 20/class
- [ ] Yard-line crossing annotation
- [ ] Sprint-model fit
- [ ] Validation vs. official 40 times
