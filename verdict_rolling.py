#!/usr/bin/env python3
"""
progressive_verdicts_print.py  (no dotenv)

- Reads summary.csv (start,end,class1,class2,class3,...)
- For each 2s (or arbitrary) row, calls the LLM with rolling window of past rows up to that time.
- Prints while running in the form:
    Input 0-2sec <line for 0-2s>
    Agent: YES/NO <why>
    Input 0-4 sec
      - 0-2s: ...
      - 2-4s: ...
    Agent: YES/NO <why>
- Adds:
    * comments: per-row verdict
    * session_verdict: rolling verdict per row (1, 0, or 0.5)

NEW:
- Compute a clip-level decision at the end:
    label = 1 if at least 10% of session_verdict == 1 else 0
- Write output file as: <out_folder>/<file_stem>_<label>.csv
  where file_stem comes from the first non-empty entry in the input 'file' column.

Notes:
- If --out is a filename (.csv), its parent folder is used as <out_folder>.
- If --out is a folder, it's used directly.
"""

import os
import time
import argparse
from pathlib import Path
from typing import List
import pandas as pd
import sys

from prompts import PROMPT_SYSTEM_MILITARY, PROMPT_SYSTEM_SHOP, PROMPT_SYSTEM_SPORTS, PROMPT_SYSTEM_ELDERS, PROMPT_SYSTEM_URBAN

# ====== PUT YOUR OPENAI API KEY HERE ======
API_KEY = "sk-proj-ETC"
# ==========================================

# Force line-buffered, write-through stdout (helps on Windows)
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)  # py3.7+
except Exception:
    pass

def say(s: str = ""):
    """Print immediately (no buffering)."""
    sys.stdout.write(s + "\n")
    sys.stdout.flush()

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Install dependencies first: pip install openai pandas")

#PROMPT_SYSTEM = PROMPT_SYSTEM_MILITARY
PROMPT_SYSTEM = PROMPT_SYSTEM_SHOP
#PROMPT_SYSTEM = PROMPT_SYSTEM_SPORTS
#PROMPT_SYSTEM = PROMPT_SYSTEM_ELDERS
#PROMPT_SYSTEM = PROMPT_SYSTEM_URBAN

PROMPT_INSTRUCTION = "Be brief. Reply strictly as: YES - <why>  or  NO - <why>. Keep under 25 words."

def build_event_line(row, text_cols: List[str]) -> str:
    start, end = row.get("start", ""), row.get("end", "")
    try:
        span = f"{float(start):.0f}-{float(end):.0f}s"
    except Exception:
        span = f"{start}-{end}"
    parts = []
    for c in text_cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(str(row[c]).strip())
    return f"{span}: " + (" | ".join(parts) if parts else "")

def choose_text_cols(df: pd.DataFrame) -> List[str]:
    pref = [c for c in ["class1","class2","class3"] if c in df.columns]
    if pref:
        return pref
    blacklist = {"start","end","t0","t1","flags","file","filename","path","id","speaker","spk","label","comments","session_verdict"}
    cands = [c for c in df.columns if c.lower() not in blacklist and df[c].dtype == "O"]
    return cands or [c for c in df.columns if c.lower() not in {"start","end"}]

def call_llm(client: "OpenAI", model: str, lines: List[str], max_retries: int = 5) -> str:
    """Call the chat model with the rolling-window lines. (No temperature for gpt-5-nano)."""
    user_content = (
        "You are an audio surveillance agent. Audio classes and speech has been transcribed in labels together "
        "with argument and agitation/emotional flags.\n"
        "You need to decide if there is a hazardous or life threatening situation.\n"
        "Timeline (rolling window):\n" +
        "\n".join(f"- {ln}" for ln in lines) +
        "\n\n" + PROMPT_INSTRUCTION
    )
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    }
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == max_retries - 1:
                return f"ERROR: {e}"
            time.sleep(1.5 * (2 ** attempt))

def verdict_to_score(text: str) -> float:
    """Map model text to {1, 0, 0.5}."""
    if not text:
        return 0.5
    t = text.strip().upper()
    if t.startswith("YES"):
        return 1.0
    if t.startswith("NO"):
        return 0.0
    if "UNSURE" in t or "UNCERTAIN" in t or "NOT SURE" in t or "MAYBE" in t:
        return 0.5
    return 0.5

def resolve_out_folder(out_arg: str) -> Path:
    """
    If --out ends with .csv, use its parent as the folder.
    Otherwise treat --out as a folder directly.
    """
    p = Path(out_arg)
    if p.suffix.lower() == ".csv":
        return p.parent
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="summary.csv", help="Input CSV")
    ap.add_argument("--out", dest="out", default="D:/real_cases", help="Output folder OR filename (.csv)")
    ap.add_argument("--model", default="gpt-5-nano", help="OpenAI model (e.g., gpt-5-nano, gpt-4o-mini)")
    ap.add_argument("--prev-lines", type=int, default=2, help="Number of previous lines in the rolling context")
    ap.add_argument("--no-print", action="store_true", help="Disable console printing")
    args = ap.parse_args()

    if not API_KEY or API_KEY.startswith("PASTE_"):
        raise SystemExit("Paste your OpenAI API key into API_KEY at the top of the script.")

    # Load & normalize
    df = pd.read_csv(args.inp, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"start","end"}.issubset(df.columns):
        raise SystemExit("Input must have 'start' and 'end' columns.")
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"], errors="coerce")
    df = df.sort_values(["start","end"]).reset_index(drop=True)

    # Read first 'file' entry (for naming output)
    first_file_value = ""
    if "file" in df.columns:
        for v in df["file"].tolist():
            s = str(v).strip()
            if s:
                first_file_value = s
                break
    file_stem = Path(first_file_value or "session").stem

    # Choose text columns for the event lines
    text_cols = choose_text_cols(df)

    # Ensure columns exist
    if "comments" not in df.columns:
        df["comments"] = ""
    if "session_verdict" not in df.columns:
        df["session_verdict"] = 0.5  # numeric per row

    client = OpenAI(api_key=API_KEY)

    # Rolling window loop (current + prev_lines)
    for i, row in df.iterrows():
        start_idx = max(0, i - args.prev_lines)
        window = df.iloc[start_idx:i+1]
        lines = [build_event_line(window.iloc[j], text_cols) for j in range(len(window))]

        # PRINT INPUT BLOCK (window range)
        if not args.no_print:
            try:
                start_sec = int(float(window.iloc[0]["start"]))
            except Exception:
                start_sec = window.iloc[0]["start"]
            try:
                end_sec = int(float(window.iloc[-1]["end"]))
            except Exception:
                end_sec = window.iloc[-1]["end"]

            if len(lines) == 1:
                body = lines[0].split(": ", 1)[1] if ": " in lines[0] else lines[0]
                say(f"Input {start_sec}-{end_sec} sec {body}")
            else:
                say(f"Input {start_sec}-{end_sec} sec")
                for ln in lines:
                    say(f"  - {ln}")

        # Get verdict for THIS window
        verdict_text = call_llm(client, args.model, lines)
        score = verdict_to_score(verdict_text)

        # Save to row
        df.at[i, "comments"] = verdict_text
        df.at[i, "session_verdict"] = score

        if not args.no_print:
            say(f"Agent: {verdict_text}")
            say()  # blank line

    # ----- Clip-level decision -----
    # Treat values == 1 as positive bins; compute ratio
    sv_numeric = pd.to_numeric(df["session_verdict"], errors="coerce")
    positive_ratio = (sv_numeric == 1.0).mean() if len(sv_numeric) else 0.0
    clip_label = 1 if positive_ratio >= 0.10 else 0

    if not args.no_print:
        say(f"[INFO] Positive bins: {positive_ratio*100:.1f}%  -> clip label = {clip_label}")

    # ----- Output path handling (append verdict to filename) -----
    out_arg = Path(args.out)
    
    if out_arg.suffix.lower() == ".csv":
        # Exact filename given: append _<label> before extension
        out_arg.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_arg.with_name(f"{out_arg.stem}_{clip_label}{out_arg.suffix}")
    else:
        # Folder given: use <folder>/<file_stem>_<label>.csv
        out_folder = out_arg
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"{file_stem}_{clip_label}.csv"
    
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not args.no_print:
        say(f"[OK] wrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()

