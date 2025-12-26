# -*- coding: utf-8 -*-
"""
Embed subtitles (.srt) into videos (.mp4) in a folder using ffmpeg.

Usage:
    python embed_srt.py "D:\path\to\folder"
or:
    python embed_srt.py "D:\path\to\folder" --replace
"""

import argparse
import subprocess
from pathlib import Path

def embed_subtitles(video_path: Path, srt_path: Path, replace: bool = False):
    """Run ffmpeg to burn-in subtitles."""
    if not srt_path.exists():
        print(f"[WARN] Missing subtitles for: {video_path.name}")
        return False

    # Prepare output
    if replace:
        out_path = video_path.with_name(video_path.stem + "__tmp.mp4")
    else:
        out_path = video_path.with_name(video_path.stem + "_withsubs.mp4")

    print(f"[PROCESS] {video_path.name}")
    print(f"          using: {srt_path.name}")

    # Escape path for ffmpeg filter
    srt_escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(video_path),
        "-vf", f"subtitles='{srt_escaped}'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        str(out_path)
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed for {video_path.name}")
        if out_path.exists():
            out_path.unlink()
        return False

    if replace:
        try:
            video_path.unlink()
            out_path.rename(video_path)
            print(f"[OK] Replaced {video_path.name}")
        except Exception as e:
            print(f"[ERROR] Could not overwrite: {e}")
            return False
    else:
        print(f"[OK] Created {out_path.name}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Embed subtitles (.srt) into .mp4 videos using ffmpeg.")
    parser.add_argument("input_folder", help="Folder containing .mp4 and .srt files")
    parser.add_argument("--replace", action="store_true",
                        help="Replace original video (default: create _withsubs.mp4)")
    args = parser.parse_args()

    root = Path(args.input_folder)
    if not root.exists():
        print(f"[ERROR] Folder not found: {root}")
        return

    mp4_files = list(root.rglob("*.mp4"))
    if not mp4_files:
        print("[WARN] No .mp4 files found.")
        return

    count_ok, count_fail, count_skip = 0, 0, 0
    print(f"[INFO] Scanning: {root}")

    for video in mp4_files:
        srt_candidate = video.with_suffix(".srt")
        if not srt_candidate.exists():
            print(f"[WARN] No SRT next to {video.name}")
            count_skip += 1
            continue

        ok = embed_subtitles(video, srt_candidate, replace=args.replace)
        if ok:
            count_ok += 1
        else:
            count_fail += 1

    print(f"\n[SUMMARY] Done: {count_ok}   Skipped (no .srt): {count_skip}   Failed: {count_fail}")


if __name__ == "__main__":
    main()
