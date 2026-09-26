"""Download the videos in data/videos.csv at 1080p into video/.

videos.csv records the SHA-256 of the exact files the dataset was built from. YouTube
can re-encode an upload, so a mismatch is reported: the pipeline still runs, but its
output may differ slightly from the released data.
"""

import hashlib
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    videos = pd.read_csv(ROOT / "data/videos.csv")
    (ROOT / "video").mkdir(exist_ok=True)
    for v in videos.itertuples():
        out = ROOT / "video" / f"{v.video_id}.mp4"
        if not out.exists() and not fetch(v.video_id, out):
            print(v.video_id, "FAILED", flush=True)
            continue
        same = sha256(out) == v.sha256
        print(v.video_id, "ok" if same else "downloaded, but differs from the file the dataset was built from", flush=True)


if __name__ == "__main__":
    main()
