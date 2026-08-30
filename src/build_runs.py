"""Segment every video into runs and bind each run to an athlete.

Resumable: videos already present in the output are skipped, so an interrupted
pass costs only the videos it had not reached.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from read_clock import PanelReader, track_runs
from read_bib import roster, bib_for_window


def calibrate_offset(best: dict[int, float], ros: pd.DataFrame,
                     max_offset: int = 80) -> tuple[int, float, int]:
    """Solve for the bib offset of this video's position block.

    Bib numbers run across a whole broadcast group (DB covers CB then SAF), with
    each sub-position a consecutive alphabetical block. Rather than hard-coding
    each group's internal order, the offset is fitted: official times are known,
    so the correct offset is the one that lines observed run times up with them.
    """
    names = ros.player_name.tolist()
    forty = dict(zip(ros.player_name, ros.forty))
    scored = []
    for k in range(max_offset + 1):
        res, missing = [], 0
        ok = True
        for bib, clock in best.items():
            i = bib - 1 - k
            if not (0 <= i < len(names)):
                ok = False
                break
            o = forty.get(names[i])
            if pd.isna(o):
                missing += 1
            else:
                res.append(abs(clock - o))
        # Every bib must land inside the roster, and on someone who actually ran:
        # an athlete without an official 40 cannot be the one on screen. Timing
        # alone does not separate offsets when a group's times cluster tightly.
        if not ok or len(res) < 3:
            continue
        scored.append((missing * 10 + float(np.mean(res)), k, len(res)))
    if not scored:
        return 0, float("nan"), 0
    score, k, hit = min(scored)
    return k, score, hit


def process(video_id: str, meta: pd.Series, reader: PanelReader,
            video_dir: pathlib.Path, fps: float) -> list[dict]:
    path = video_dir / f"{video_id}.mp4"
    if not path.exists():
        print(f"  {video_id}: no local file, skipped", file=sys.stderr)
        return []

    positions = meta.positions.split("|")
    ros = roster(int(meta.season), positions)
    official = dict(zip(ros.player_name, ros.forty))

    runs = track_runs(str(path), float(meta.duration), reader, fps=fps)

    seg = []
    for c, complete in runs:
        t0, t1 = c[0][0], c[-1][0]
        bib, agree = bib_for_window(str(path), t0, min(t1, t0 + 6), reader)
        seg.append((c, bib, agree, complete))

    # Only completed runs calibrate the offset; a fragment's final value is not a
    # 40 time and would drag the fit.
    best: dict[int, float] = {}
    for c, bib, _, complete in seg:
        if bib is not None and complete:
            best[bib] = min(best.get(bib, 9.9), c[-1][1])
    offset, resid, _ = calibrate_offset(best, ros)
    names = ros.player_name.tolist()
    print(f"     bib offset {offset} (median |clock-official| {resid:.3f})",
          file=sys.stderr)

    rows = []
    for c, bib, agree, complete in seg:
        t0, t1 = c[0][0], c[-1][0]
        i = (bib - 1 - offset) if bib is not None else -1
        name = names[i] if 0 <= i < len(names) else None
        # Clock zero in video time, from the actively-running samples only:
        # leading frames parked at 0.00 would bias the intercept.
        arr = np.array([(t, v) for t, v in c if v > 0.05])
        t_zero = float(np.mean(arr[:, 0] - arr[:, 1])) if len(arr) else t0
        rows.append({
            "video_id": video_id, "season": int(meta.season), "group": meta.group,
            "cls": meta.cls, "bib": bib, "player_name": name,
            "bib_agreement": round(agree, 2),
            "t_zero": round(t_zero, 3),
            "t_start": round(t0, 2), "t_end": round(t1, 2),
            "final_clock": c[-1][1], "n_samples": len(c),
            "official_forty": official.get(name) if name else None,
            "bib_offset": offset,
            "complete": complete,
        })
    return rows


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag runs for athletes that a second video repeats.

    Consecutive group uploads overlap at their seam: the next video opens by
    replaying the athlete the previous one closed on. The replay is the copy whose
    bib sits at the very start of its video's bib range, so the other video's runs
    are kept.
    """
    df = df.copy()
    df["duplicate"] = False
    if "player_name" not in df or df.empty:
        return df
    lo = df.groupby("video_id").bib.transform("min")
    for name, g in df[df.player_name.notna()].groupby("player_name"):
        if g.video_id.nunique() < 2:
            continue
        lead_in = g.index[(g.bib == lo[g.index]) & (g.video_id != g.video_id.iloc[0])]
        keep_vid = g.loc[~g.index.isin(lead_in), "video_id"]
        if keep_vid.empty or lead_in.empty:
            continue
        df.loc[lead_in, "duplicate"] = True
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/videos_checked.csv")
    ap.add_argument("--out", default="data/runs.csv")
    ap.add_argument("--video-dir", default="video")
    ap.add_argument("--templates", default="data/glyph_templates.npz")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--only", nargs="*", help="limit to these video ids")
    args = ap.parse_args()

    vids = pd.read_csv(args.videos, dtype={"video_id": str})
    vids = vids[vids.duration.notna()]
    if args.only:
        vids = vids[vids.video_id.isin(args.only)]

    out = pathlib.Path(args.out)
    done, prev = set(), []
    if out.exists():
        old = pd.read_csv(out, dtype={"video_id": str})
        done = set(old.video_id)
        prev = old.to_dict("records")

    reader = PanelReader(args.templates)
    rows = list(prev)
    for _, meta in vids.iterrows():
        if meta.video_id in done:
            print(f"  {meta.video_id}: already done", file=sys.stderr)
            continue
        print(f"  {meta.video_id}: {meta.group}", file=sys.stderr)
        new = process(meta.video_id, meta, reader,
                      pathlib.Path(args.video_dir), args.fps)
        rows += new
        pd.DataFrame(rows).to_csv(out, index=False)
        named = sum(1 for r in new if r["player_name"])
        print(f"     {len(new)} runs, {named} identified", file=sys.stderr)

    df = mark_duplicates(pd.DataFrame(rows))
    df.to_csv(out, index=False)
    print(f"\n{len(df)} runs across {df.video_id.nunique()} videos -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
