#!/usr/bin/env python3
"""
csv_summary_to_srt.py

Convert a summary CSV (with columns: file, start, end, comments) into an SRT file.
The output filename is derived from the FIRST value in the 'file' column:
    <basename_of_first_file>.srt

Usage:
    python csv_summary_to_srt.py summary_with_comments.csv
    # optional explicit out path (overrides auto naming):
    python csv_summary_to_srt.py summary_with_comments.csv --out myvideo.srt
"""

import argparse
import math
import os
from pathlib import Path
import pandas as pd

def sec_to_srt_timestamp(sec: float) -> str:
    """Convert seconds (float) to SRT timestamp 'HH:MM:SS,mmm'."""
    if pd.isna(sec):
        sec = 0.0
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    sec -= h * 3600
    m = int(sec // 60)
    sec -= m * 60
    s = int(sec)
    ms = int(round((sec - s) * 1000))
    # handle rounding overflow (e.g., 59.9995 -> next second)
    if ms >= 1000:
        s += 1
        ms -= 1000
    if s >= 60:
        m += 1
        s -= 60
    if m >= 60:
        h += 1
        m -= 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def choose_output_name(df: pd.DataFrame, out_arg: str | None) -> Path:
    if out_arg:
        p = Path(out_arg)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    if "file" not in df.columns or df["file"].dropna().empty:
        # fallback if file column missing/empty
        return Path("output.srt")
    first_file = str(df["file"].dropna().iloc[0]).strip()
    # derive base name (remove folders), strip extension, add .srt
    base = os.path.splitext(os.path.basename(first_file))[0]
    if not base:
        base = "output"
    return Path(f"{base}.srt")

def main():
    ap = argparse.ArgumentParser(description="Make an SRT from a summary CSV (file,start,end,comments).")
    ap.add_argument("csv", help="Input CSV (e.g., summary_with_comments_c_m_2.csv)")
    ap.add_argument("--out", help="Optional explicit SRT path (e.g., D:/clips/video.srt)")
    args = ap.parse_args()

    # Read CSV (robust to BOM; keep strings as-is)
    df = pd.read_csv(args.csv, encoding="utf-8-sig", keep_default_na=False)
    # Normalize headers to lower for safety
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"start", "end", "comments"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # Sort by start to ensure chronological SRT
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce")
    df = df.dropna(subset=["start","end"]).sort_values(["start","end"]).reset_index(drop=True)

    # Build SRT cues, skipping empty comments
    cues = []
    for _, row in df.iterrows():
        txt = str(row.get("comments", "")).strip()
        if not txt:
            continue
        t0 = max(0.0, float(row["start"]))
        t1 = max(t0, float(row["end"]))  # ensure non-negative duration
        cues.append((t0, t1, txt))

    if not cues:
        print("[WARN] No non-empty 'comments' found; writing an empty SRT.")
    
    out_path = choose_output_name(df, args.out)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (t0, t1, txt) in enumerate(cues, start=1):
            f.write(f"{i}\n")
            f.write(f"{sec_to_srt_timestamp(t0)} --> {sec_to_srt_timestamp(t1)}\n")
            # SRT supports multi-line; keep as single line for now
            f.write(f"{txt}\n\n")

    print(f"[OK] wrote {out_path.resolve()}")
    print("Tip: For auto-load in most players, ensure your VIDEO file shares the same basename as the SRT.")

if __name__ == "__main__":
    main()
