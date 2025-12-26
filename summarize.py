#!/usr/bin/env python3
"""
summarize.py — Merge ASR + CLAP/Speakers/Prosody/Pitch flags into AST cells that contain:
               speech / monologue / conversation (exact words, case-insensitive).

Robustness:
- If any input CSV (ASR, CLAP, SPK, PROSODY, PITCH) is missing/empty/malformed, it is skipped.
- Only available information is integrated; no crashes.

Output:
  --out summary.csv  (default)
"""

import argparse
import os
import re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# ---------- configuration ----------
SPEECH_TRIGGERS = re.compile(r"\b(?:speech|monologue|conversation)\b", re.IGNORECASE)

# Treat these as "empty" tokens (won't be emitted as flags)
BAD_FLAG_STRINGS = {"", "nan", "none", "null", "na", "n/a"}

# Match bracket groups like [FLAG]
BRACKET_GROUPS = re.compile(r"\[[^\]]*\]")

# ---------- helpers ----------
def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure float 'start','end' columns exist (accept 't0','t1' too)."""
    if df.empty:
        return df
    out = df.copy()
    cols_l = {c.lower().strip(): c for c in out.columns}
    # rename to start/end when possible
    if ("start" not in cols_l or "end" not in cols_l) and ("t0" in cols_l and "t1" in cols_l):
        out = out.rename(columns={cols_l["t0"]: "start", cols_l["t1"]: "end"})
    else:
        if "start" in cols_l: out = out.rename(columns={cols_l["start"]: "start"})
        if "end"   in cols_l: out = out.rename(columns={cols_l["end"]: "end"})
    if "start" not in out.columns or "end" not in out.columns:
        raise ValueError("Expected time columns 'start'/'end' or 't0'/'t1'.")
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"]   = pd.to_numeric(out["end"],   errors="coerce")
    out = out.dropna(subset=["start","end"]).reset_index(drop=True)
    return out

def is_bad_token(s: str) -> bool:
    return (s is None) or (str(s).strip().lower() in BAD_FLAG_STRINGS)

def split_flags(cell) -> List[str]:
    """Parse 'A;B' or 'A B' into tokens; if already bracketed, strip brackets first."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if is_bad_token(s):
        return []
    if "[" in s and "]" in s:
        parts = [p for p in s.replace("][", "|").replace("[", "").replace("]", "").split("|")]
        return [p.strip() for p in parts if not is_bad_token(p)]
    parts = [p.strip() for p in (s.split(";") if ";" in s else s.split())]
    return [p for p in parts if not is_bad_token(p)]

def extract_bracket_groups(s: str) -> List[str]:
    """Return sanitized bracket groups like ['[FLAG]'] from a string."""
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for g in BRACKET_GROUPS.findall(s):
        inner = g[1:-1].strip()
        if not is_bad_token(inner):
            out.append(f"[{inner}]")
    return out

def bracket_join(tokens: List[str]) -> str:
    """Make '[tok1][tok2]' from sanitized tokens."""
    return "".join(f"[{t}]" for t in tokens if not is_bad_token(t))

def df_safe_read(path: str) -> pd.DataFrame:
    """
    Read CSV with keep_default_na=False so strings like 'NA'/'null' stay literal.
    Returns empty DataFrame if path missing/unreadable.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    for enc in ("utf-8-sig", None):
        try:
            return pd.read_csv(path, encoding=enc, keep_default_na=False)
        except Exception:
            continue
    return pd.DataFrame()

def guess_asr_text_cols(asr_df: pd.DataFrame) -> List[str]:
    """Priority-ordered list of plausible ASR text columns."""
    prefs = ["text", "transcript", "asr", "content", "hypothesis", "class1", "class2", "class3"]
    cols = [c.lower() for c in asr_df.columns]
    chosen = [c for c in prefs if c in cols]
    if chosen:
        return chosen
    blacklist = {"start","end","t0","t1","flags","flags_detailed","file","filename","path","id","speaker","spk","label"}
    return [c for c in asr_df.columns if asr_df[c].dtype == "O" and c.lower() not in blacklist]

# ---------- per-stream collectors ----------
def collect_asr_text(asr_df: pd.DataFrame, s0: float, s1: float, text_cols: List[str]) -> str:
    if asr_df.empty or not text_cols:
        return ""
    tmp = asr_df.copy()
    try:
        tmp["ov"] = tmp.apply(lambda r: interval_overlap(s0, s1, float(r["start"]), float(r["end"])), axis=1)
    except Exception:
        return ""
    tmp = tmp[tmp["ov"] > 0].sort_values(["start", "end"]).reset_index(drop=True)
    if tmp.empty:
        return ""
    # try lowercased match first
    lower_cols = {c.lower(): c for c in tmp.columns}
    col = next((lower_cols[c] for c in text_cols if c in lower_cols), None)
    if not col:
        # fall back to exact names as-is
        for c in text_cols:
            if c in tmp.columns:
                col = c
                break
    if not col:
        return ""
    texts = [str(x).strip() for x in tmp[col].astype(str).tolist() if not is_bad_token(x)]
    out, last = [], None
    for t in texts:
        if t != last:
            out.append(t)
            last = t
    return " ".join(out).strip()

def collect_overlapping_clap_detailed(clap_df: pd.DataFrame, s0: float, s1: float) -> str:
    """Use CLAP 'flags_detailed' if present; fallback to 'flags' tokens. Dedupe & preserve order."""
    if clap_df.empty:
        return ""
    tmp = clap_df.copy()
    try:
        tmp["ov"] = tmp.apply(lambda r: interval_overlap(s0, s1, float(r["start"]), float(r["end"])), axis=1)
    except Exception:
        return ""
    tmp = tmp[tmp["ov"] > 0].sort_values(["start","end"]).reset_index(drop=True)
    seen = set()
    ordered = []
    for _, r in tmp.iterrows():
        fd = str(r.get("flags_detailed", "") or "")
        groups = extract_bracket_groups(fd)
        if groups:
            for g in groups:
                if g not in seen:
                    seen.add(g); ordered.append(g)
        else:
            for t in split_flags(r.get("flags", "")):
                g = f"[{t}]"
                if g not in seen:
                    seen.add(g); ordered.append(g)
    return "".join(ordered)

def collect_overlapping_speaker_flags(spk_df: pd.DataFrame, s0: float, s1: float) -> List[str]:
    """Collect speaker flags tokens (e.g., HIGH_OVERLAP;RAPID_TURNS) over [s0,s1]."""
    if spk_df.empty:
        return []
    tmp = spk_df.copy()
    try:
        tmp["ov"] = tmp.apply(lambda r: interval_overlap(s0, s1, float(r["start"]), float(r["end"])), axis=1)
    except Exception:
        return []
    tmp = tmp[tmp["ov"] > 0].sort_values(["start","end"]).reset_index(drop=True)
    seen = set()
    ordered = []
    for _, r in tmp.iterrows():
        for tok in split_flags(r.get("flags", "")):
            if tok not in seen:
                seen.add(tok); ordered.append(tok)
    return ordered

def collect_overlapping_bracketed_any(df: pd.DataFrame, s0: float, s1: float, preferred_cols: List[str]) -> str:
    """
    For streams like prosody/pitch:
      - prefer a 'flags' column; fallback to 'class1'
      - accept bracketed groups or plain tokens; sanitize/dedupe.
    """
    if df.empty:
        return ""
    tmp = df.copy()
    try:
        tmp["ov"] = tmp.apply(lambda r: interval_overlap(s0, s1, float(r["start"]), float(r["end"])), axis=1)
    except Exception:
        return ""
    tmp = tmp[tmp["ov"] > 0].sort_values(["start","end"]).reset_index(drop=True)
    if tmp.empty:
        return ""
    # try lowercased match first
    lower_cols = {c.lower(): c for c in tmp.columns}
    col = next((lower_cols[c] for c in preferred_cols if c in lower_cols), None)
    if not col:
        for c in preferred_cols:
            if c in tmp.columns:
                col = c; break
    if not col:
        return ""
    seen = set()
    ordered = []
    for _, r in tmp.iterrows():
        raw = str(r.get(col, "") or "")
        groups = extract_bracket_groups(raw)
        if groups:
            for g in groups:
                if g not in seen:
                    seen.add(g); ordered.append(g)
        else:
            for t in split_flags(raw):
                g = f"[{t}]"
                if g not in seen:
                    seen.add(g); ordered.append(g)
    return "".join(ordered)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Replace AST 'speech/monologue/conversation' cells with ASR text + flags (CLAP/Speakers/Prosody/Pitch)."
    )
    ap.add_argument("--ast",     default="streaming_AST.csv",        help="AST CSV (start,end,class1,...) [required]")
    ap.add_argument("--asr",     default="streaming_ASR.csv",        help="ASR CSV (start/end or t0/t1 + text)")
    ap.add_argument("--clap",    default="streaming_CLAP.csv",       help="CLAP CSV (start/end or t0/t1 + flags_detailed/flags)")
    ap.add_argument("--spk",     default="streaming_speakers.csv",   help="Speakers CSV (start/end or t0/t1 + flags)")
    ap.add_argument("--prosody", default="streaming_prosody.csv",    help="Prosody CSV (start/end or t0/t1 + flags or class1)")
    ap.add_argument("--pitch",   default="streaming_pitch.csv",      help="Pitch CSV (start/end or t0/t1 + flags or class1)")
    ap.add_argument("--out",     default="summary.csv",              help="Output CSV")
    args = ap.parse_args()

    # ---------- AST (required) ----------
    ast = df_safe_read(args.ast)
    if ast.empty:
        raise SystemExit(f"[ERROR] AST CSV not found or empty: {args.ast}")
    ast.columns = [c.strip().lower() for c in ast.columns]
    try:
        ast = normalize_time_columns(ast)
    except Exception as e:
        raise SystemExit(f"[ERROR] AST CSV must have 'start/end' or 't0/t1': {e}")
    ast = ast.sort_values(["start","end"]).reset_index(drop=True)
    class_cols = [c for c in ast.columns if c.startswith("class")]
    if not class_cols:
        raise SystemExit("[ERROR] AST must include at least one class* column (e.g., class1).")

    # ---------- ASR (optional) ----------
    asr = df_safe_read(args.asr)
    if not asr.empty:
        asr.columns = [c.strip().lower() for c in asr.columns]
        try:
            asr = normalize_time_columns(asr).sort_values(["start","end"]).reset_index(drop=True)
        except Exception:
            asr = pd.DataFrame()
    asr_text_candidates = guess_asr_text_cols(asr) if not asr.empty else []

    # ---------- CLAP (optional; prefer flags_detailed) ----------
    clap = df_safe_read(args.clap)
    if not clap.empty:
        clap.columns = [c.strip().lower() for c in clap.columns]
        try:
            clap = normalize_time_columns(clap)
        except Exception:
            clap = pd.DataFrame()
    if not clap.empty:
        if "flags_detailed" not in clap.columns:
            clap["flags_detailed"] = ""
        keep_cols = ["start","end","flags_detailed"] + (["flags"] if "flags" in clap.columns else [])
        clap = clap[[c for c in keep_cols if c in clap.columns]].sort_values(["start","end"]).reset_index(drop=True)

    # ---------- Speakers (optional) ----------
    spk = df_safe_read(args.spk)
    if not spk.empty:
        spk.columns = [c.strip().lower() for c in spk.columns]
        try:
            spk = normalize_time_columns(spk)
        except Exception:
            spk = pd.DataFrame()
    if not spk.empty:
        if "flags" not in spk.columns:
            spk["flags"] = ""
        spk = spk[[c for c in ["start","end","flags"] if c in spk.columns]] \
               .sort_values(["start","end"]).reset_index(drop=True)

    # ---------- Prosody (optional) ----------
    pros = df_safe_read(args.prosody)
    if not pros.empty:
        pros.columns = [c.strip().lower() for c in pros.columns]
        try:
            pros = normalize_time_columns(pros).sort_values(["start","end"]).reset_index(drop=True)
        except Exception:
            pros = pd.DataFrame()

    # ---------- Pitch (optional) ----------
    pit = df_safe_read(args.pitch)
    if not pit.empty:
        pit.columns = [c.strip().lower() for c in pit.columns]
        try:
            pit = normalize_time_columns(pit).sort_values(["start","end"]).reset_index(drop=True)
        except Exception:
            pit = pd.DataFrame()

    # ----- Integrate -----
    for i, row in ast.iterrows():
        s0, s1 = float(row["start"]), float(row["end"])

        # Collect flags across available streams
        clap_str = collect_overlapping_clap_detailed(clap, s0, s1) if not clap.empty else ""
        spk_tokens = collect_overlapping_speaker_flags(spk, s0, s1) if not spk.empty else []
        spk_str = bracket_join(spk_tokens) if spk_tokens else ""
        pros_str = collect_overlapping_bracketed_any(pros, s0, s1, preferred_cols=["flags", "class1"]) if not pros.empty else ""
        pit_str  = collect_overlapping_bracketed_any(pit,  s0, s1, preferred_cols=["flags", "class1"]) if not pit.empty else ""

        flags_str = "".join([x for x in (clap_str, spk_str, pros_str, pit_str) if x])

        # Replace only exact triggers in class* cells
        for cc in class_cols:
            txt_raw = row.get(cc, "")
            if pd.isna(txt_raw):
                continue
            txt = str(txt_raw)

            if SPEECH_TRIGGERS.search(txt):
                asr_text = collect_asr_text(asr, s0, s1, asr_text_candidates) if asr_text_candidates else ""
                if asr_text:
                    new_cell = asr_text if not flags_str else f"{asr_text} {flags_str}"
                else:
                    new_cell = txt if not flags_str else f"{txt} {flags_str}"
                ast.at[i, cc] = new_cell

    # ----- De-duplicate within each row across class1/class2/class3 -----
    target_cols = [c for c in ["class1","class2","class3"] if c in ast.columns]
    if target_cols:
        for ridx, r in ast.iterrows():
            seen = set()
            for cc in target_cols:
                v = r.get(cc, None)
                if pd.isna(v):
                    continue
                s = str(v)
                if s in seen:
                    ast.at[ridx, cc] = ""
                else:
                    seen.add(s)

    # Save
    out_path = Path(args.out)
    ast.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out_path.resolve()}")

if __name__ == "__main__":
    main()
