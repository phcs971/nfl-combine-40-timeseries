"""Dump a contact sheet and sample frames from a local video, for structure inspection."""

import argparse
import pathlib
import subprocess
import sys


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr)
        raise SystemExit(f"ffmpeg failed: {' '.join(cmd[:6])}...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=10.0, help="seconds between tiles")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--outdir", default="frames")
    ap.add_argument("--at", type=float, nargs="*", default=[],
                    help="also dump full-res stills at these timestamps")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(args.video).stem

    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", args.video],
        capture_output=True, text=True).stdout.strip())
    n = int(dur // args.every)
    rows = -(-n // args.cols)

    sheet = out / f"{stem}_sheet.jpg"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video,
         "-vf", f"fps=1/{args.every},scale={args.width}:-1,"
                f"tile={args.cols}x{rows}",
         "-frames:v", "1", "-q:v", "3", str(sheet)])
    print(f"{dur:.0f}s -> {n} tiles ({args.cols}x{rows})  {sheet}")

    for t in args.at:
        still = out / f"{stem}_t{t:g}.jpg"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t),
             "-i", args.video, "-frames:v", "1", "-q:v", "2", str(still)])
        print(f"  still @ {t:g}s  {still}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
