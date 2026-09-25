"""Download the videos in data/videos.csv at 1080p into video/."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "bv*[height=1080][ext=mp4]/bv*[height<=1080]"
# YouTube intermittently 403s the default client; fall back through others.
CLIENTS = [None, "tv", "web_safari", "mweb", "ios"]


def fetch(video_id: str, out: Path) -> bool:
    for client in CLIENTS:
        cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings",
               "-f", FORMAT, "-o", str(out), "--", video_id]
        if client:
            cmd[3:3] = ["--extractor-args", f"youtube:player_client={client}"]
        if subprocess.run(cmd).returncode == 0 and out.exists():
            return True
    return False


def main() -> None:
    videos = pd.read_csv(ROOT / "data/videos.csv")
    (ROOT / "video").mkdir(exist_ok=True)
    for vid in videos.video_id:
        out = ROOT / "video" / f"{vid}.mp4"
        if out.exists():
            continue
        print(vid, "ok" if fetch(vid, out) else "FAILED", flush=True)


if __name__ == "__main__":
    main()
