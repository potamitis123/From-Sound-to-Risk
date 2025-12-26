#!/usr/bin/env python3
"""
progressive_verdicts_print.py  (Ollama/Llama 3 edition + Full Output + Timing)

- Reads summary.csv
- Uses local Llama 3 via Ollama.
- OUTPUTS:
    1. The input lines read from summary.csv (Context)
    2. The Agent's Verdict (YES/NO)
    3. The time taken to process that specific row.
"""

import os
import time
import argparse
from pathlib import Path
from typing import List
import pandas as pd
import sys

# Ensure you have the prompts.py file in the same directory or python path
from prompts import PROMPT_SYSTEM_MILITARY, PROMPT_SYSTEM_SHOP, PROMPT_SYSTEM_SPORTS, PROMPT_SYSTEM_ELDERS, PROMPT_SYSTEM_URBAN

# Force line-buffered, write-through stdout (helps on Windows)
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
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

# Default Prompt System
PROMPT_SYSTEM = PROMPT_SYSTEM_SHOP
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
    """Call the local Llama model with the rolling-window lines."""
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
            response = client.chat.completions.create(**kwargs)
            
            # Robust parsing Logic
            if response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            if hasattr(response.choices[0], "delta") and response.choices[0].delta.content:
                return response.choices[0].delta.content.strip()
            if hasattr(response.choices[0], "text") and response.choices[0].text:
                return response.choices[0].text.strip()
            return "(no content returned)"
            
        except Exception as e:
            if attempt == max_retries - 1:
                return f"ERROR: {e}"
            time.sleep(1.5 * (2 ** attempt))

def verdict_to_score(text: str) -> float:
    if not text: return 0.5
    t = text.strip().upper()
    if t.startswith("YES"): return 1.0
    if t.startswith("NO"): return 0.0
    if "UNSURE" in t or "UNCERTAIN" in t or "NOT SURE" in t or "MAYBE" in t: return 0.5
    return 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="summary.csv", help="Input CSV")
    ap.add_argument("--out", dest="out", default="D:/real_cases", help="Output folder OR filename (.csv)")
    ap.add_argument("--model", default="llama3.2:latest", help="Ollama model name")
    ap.add_argument("--prev-lines", type=int, default=2, help="Lines in rolling context")
    ap.add_argument("--no-print", action="store_true", help="Disable console printing")
    args = ap.parse_args()

    # Load & normalize
    df = pd.read_csv(args.inp, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"start","end"}.issubset(df.columns):
        raise SystemExit("Input must have 'start' and 'end' columns.")
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"], errors="coerce")
    df = df.sort_values(["start","end"]).reset_index(drop=True)

    # File stem for output naming
    first_file_value = ""
    if "file" in df.columns:
        for v in df["file"].tolist():
            s = str(v).strip()
            if s:
                first_file_value = s
                break
    file_stem = Path(first_file_value or "session").stem

    text_cols = choose_text_cols(df)
    if "comments" not in df.columns: df["comments"] = ""
    if "session_verdict" not in df.columns: df["session_verdict"] = 0.5

    # Initialize Local Ollama Client
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    say(f"Processing '{args.inp}' with model '{args.model}'...\n")

    # Rolling window loop
    for i, row in df.iterrows():
        # Start Timer
        t0 = time.time()

        start_idx = max(0, i - args.prev_lines)
        window = df.iloc[start_idx:i+1]
        lines = [build_event_line(window.iloc[j], text_cols) for j in range(len(window))]

        # --- A) PRINT INPUT BLOCK (mimic verdict_rolling.py) ---
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

        # Call LLM
        verdict_text = call_llm(client, args.model, lines)
        score = verdict_to_score(verdict_text)

        # Save to row
        df.at[i, "comments"] = verdict_text
        df.at[i, "session_verdict"] = score

        # End Timer
        elapsed = time.time() - t0

        # --- B & C) PRINT OUTPUT + TIME ---
        if not args.no_print:
            say(f"Agent: {verdict_text}")
            say(f"   -> Time: {elapsed:.4f} sec")
            say() # blank line separator

    # Clip-level decision
    sv_numeric = pd.to_numeric(df["session_verdict"], errors="coerce")
    positive_ratio = (sv_numeric == 1.0).mean() if len(sv_numeric) else 0.0
    clip_label = 1 if positive_ratio >= 0.10 else 0

    if not args.no_print:
        say(f"[INFO] Positive bins: {positive_ratio*100:.1f}%  -> clip label = {clip_label}")

    # Output path handling
    out_arg = Path(args.out)
    if out_arg.suffix.lower() == ".csv":
        out_arg.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_arg.with_name(f"{out_arg.stem}_{clip_label}{out_arg.suffix}")
    else:
        out_folder = out_arg
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"{file_stem}_{clip_label}.csv"
    
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not args.no_print:
        say(f"[OK] wrote {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()