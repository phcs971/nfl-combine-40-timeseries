"""Validate the video registry against live YouTube metadata."""

import argparse
import json
import subprocess
import sys

import pandas as pd

FIELDS = "%(id)s\t%(duration)s\t%(fps)s\t%(title)s"


def probe(video_id: str) -> dict | None:
    # Full URL, not a bare id: several combine ids start with "-" and would be
    # parsed as a flag.
    url = f"https://www.youtube.com/watch?v={video_id}"
    r = subprocess.run(
        ["yt-dlp", "--skip-download", "--no-warnings", "--print", FIELDS, url],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    vid, dur, fps, title = r.stdout.strip().split("\t", 3)
    return {"duration": float(dur), "fps": float(fps), "title": title}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="data/videos.csv")
    ap.add_argument("--out", default="data/videos_checked.csv")
    args = ap.parse_args()

    reg = pd.read_csv(args.registry, dtype={"video_id": str})
    rows, dead = [], []
    for vid in reg.video_id:
        meta = probe(vid)
        if meta is None:
            dead.append(vid)
            rows.append({"duration": None, "fps": None, "title": None})
        else:
            rows.append(meta)
        print(f"  {vid}  {'OK' if meta else 'UNAVAILABLE'}", file=sys.stderr)

    out = pd.concat([reg, pd.DataFrame(rows)], axis=1)
    out.to_csv(args.out, index=False)

    live = out[out.duration.notna()]
    print(f"\n{len(live)}/{len(out)} available -> {args.out}")
    print(f"total footage: {live.duration.sum() / 3600:.2f} h")
    print(f"frame rates present: {sorted(live.fps.unique())}")
    if dead:
        print(f"UNAVAILABLE: {dead}")
    for cls, g in live.groupby("cls"):
        print(f"  {cls:6} {len(g):2d} videos  {g.duration.sum()/3600:.2f} h")
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
