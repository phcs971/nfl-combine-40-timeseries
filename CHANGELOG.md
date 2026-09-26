# Changelog

Versions follow [semantic versioning](https://semver.org/) for the data: a major
version changes values or columns in a way that breaks comparisons with earlier
releases; a minor version adds runs or columns; a patch fixes documentation.

## 1.0.1 — 2026-09-27

Documentation only; the data files are identical to 1.0.0.

- Archived on Zenodo, which mints the dataset's DOI from this release.
- Author ORCID and contact email in `CITATION.cff`, the datasheet and the README.

## 1.0.0 — 2026-09-27

First public release.

- 276 runs (164 athletes) kept from 326 detected in 11 NFL YouTube videos; 48
  rejected by quality control, 2 excluded (a QB in the WR group 2 session).
- Per-frame series from one frame before the clock start to one frame after the
  stop, plus a 101-point time grid and a 0–39 yd distance grid.
- `runs.csv` carries the unofficial broadcast 40-yd time (`time_40yd`) and the
  line groups' 10-yd split (`time_10yd`).
- Lane scale calibrated on the painted yard lines: 1.8 m per dash period.
- Machine-readable data dictionary (`datapackage.json`), datasheet, licences and
  citation file.

### Changes from the pre-release commits on `main`

These commits (`c67e064`, `2038408`) were never tagged; values differ from 1.0.0.

- `c67e064` assumed 2.0 yd per dash; positions ran ~2% long (0.9 yd at 40 yd).
- `2038408` fixed the scale and took the cross-lane direction from the yard lines.
- 1.0.0 fixes a clock-reading bug: the panel field holding the clock jitters by a
  few pixels, and fixed-width binning split it in two. That dropped 23 runs entirely
  and stopped 6 runs' clocks early (e.g. 3.94 s instead of 4.42 s). Run ids were
  renumbered chronologically as a result.
- 1.0.0 renames `final_time` / `split10` to `time_40yd` / `time_10yd`.
- 1.0.0 removes the broadcast-graphic templates from the repository; they are rebuilt
  locally from the committed labels (`data/templates/`).
