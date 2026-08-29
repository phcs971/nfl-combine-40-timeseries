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

**Video availability drives selection, not the reverse.** Official footage is the
scarce resource and it correlates with being fast and drafted. Athletes are sampled
*within* the set that has video, so the bias is measurable rather than hidden.

**Draft status is rebuilt, never read directly.** The nflverse combine release ships
`draft_round` unpopulated for the most recent class — all 319 of the 2026 rows are
null. Taken at face value that reads as "everyone went undrafted." Status is joined
from the `draft_picks` release on `pfr_id`, with a name-normalized fallback for the
~8% of rows carrying no `pfr_id`.

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

## Known error sources

- **Slow-motion replays.** Combine broadcasts cut to slo-mo constantly, and a replay
  is frame-indistinguishable from a live run. Every clip needs explicit real-time
  verification.
- **Container frame rate.** YouTube re-encodes; a 60 fps broadcast may arrive at
  30 fps. Actual fps is recorded per clip, never assumed.
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
uv run python src/build_frame.py --seasons 2025 2026
```

Writes `data/frame.csv`, the sampling frame of 380 athletes.

## Status

- [x] 1 — Sampling frame from combine + draft data
- [ ] 2 — Video inventory (enumerate available official footage)
- [ ] 3 — Stratified sample, 20/class, validated against population
- [ ] 4 — Frame extraction, fps + real-time verification
- [ ] 5 — Yard-line crossing annotation
- [ ] 6 — Sprint-model fit → position/velocity/acceleration series
- [ ] 7 — Validation: derived vs. official 40 times

Code is MIT licensed. Derived measurements are released under CC BY 4.0.
