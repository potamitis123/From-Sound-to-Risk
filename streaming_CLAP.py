#!/usr/bin/env python3
"""
streaming_CLAP.py
-----------------
Zero-shot CLAP scoring in fixed windows with LLM-friendly flags.

CSV columns:
    file,t0,t1,flags,flags_detailed,description,neutral_max,
    scores_<flag>,top_phrase_<flag> ...
"""

# ---- suppress noisy warnings (safe) ----
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.set_audio_backend has been deprecated.*",
    category=UserWarning,
    module=r"torch_audiomentations\.utils\.io"
)

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sys

# Primary + fallback loaders
import soundfile as sf             # fast, but strict about formats/paths
import librosa                     # robust fallback for many formats

from transformers import pipeline  # zero-shot audio classification (CLAP models)

# --------------------------
# Flag taxonomy (phrases per flag)
# --------------------------
FLAG_TAXONOMY: Dict[str, List[str]] = {
    "stress": [
        "stressed breathing", "incoherent wording", "tense voice",
        "shaky voice", "panicked breathing", "hyperventilation",
        "urgent whisper", "raised voice",
    ],
    "pain": [
        "crying in pain", "groaning in pain", "pain scream", "moaning from pain",
    ],
    "despair": [
        "desperate crying", "sobbing in despair", "pleading voice", "hopeless crying",
    ],
    "cry": ["crying", "weeping", "sobbing", "whimpering"],
    "agony": ["agonizing scream", "screaming in agony", "moaning in agony"],
    "high_agitation": [
        "aggressive shouting", "angry yelling", "frantic shouting",
        "panicked screaming", "arguing loudly", "raised voice", "fear"
    ],
}

# Neutral/distractor labels for calibration
NEUTRALS = [
    "calm talking", "neutral conversation", "polite conversation",
    "background noise", "silence", "soft music", "traffic ambience",
]

HYPOTHESIS_TEMPLATE = "This audio expresses {}."
MODEL_ID = "laion/larger_clap_music_and_speech"

# --------------------------
# Reason canonicalization for cleaner flags_detailed
# --------------------------
REASON_CANON = {
    "aggressive shouting": "shouting",
    "angry yelling": "shouting",
    "frantic shouting": "shouting",
    "panicked screaming": "panicked_screaming",
    "raised voice": "raised_voice",
    "tense voice": "tense_voice",
    "shaky voice": "shaky_voice",
    "stressed breathing": "stressed_breathing",
    "hyperventilation": "hyperventilation",
    "urgent whisper": "urgent_whisper",
    "moaning from pain": "moaning_pain",
    "crying in pain": "crying_pain",
    "pain scream": "pain_scream",
    "groaning in pain": "groaning_pain",
    "desperate crying": "desperate_crying",
    "sobbing in despair": "sobbing_despair",
    "pleading voice": "pleading_voice",
    "hopeless crying": "hopeless_crying",
}
def canon_reason(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return REASON_CANON.get(s, s.replace(" ", "_").replace("/", "_").lower())

# --------------------------
# Audio helpers (robust)
# --------------------------
def _normalize_path(p: str) -> str:
    """Expand ~, strip quotes, and normalize separators."""
    if p is None:
        return ""
    p = p.strip().strip('"').strip("'")
    return os.path.normpath(os.path.expanduser(p))

def _load_with_soundfile(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)

def _load_with_librosa(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), int(target_sr)

def load_audio(path: str, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """Try soundfile first, fallback to librosa; resample to 48k; ensure mono."""
    path = _normalize_path(path)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Audio path invalid or not found: {repr(path)}")
    try:
        wav, sr = _load_with_soundfile(path)
    except Exception as e_sf:
        try:
            wav, sr = _load_with_librosa(path, target_sr)
        except Exception as e_lr:
            raise RuntimeError(
                f"Failed to load with soundfile ({e_sf}) and librosa ({e_lr}). Path={repr(path)}"
            )
    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.float32)
    else:
        wav = wav.astype(np.float32)
    return wav, sr

def frame_iter(sig: np.ndarray, sr: int, win_s: float, hop_s: float):
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    n = len(sig); t = 0
    while t < n:
        t1 = min(n, t + win)
        yield (t, t1, sig[t:t1])
        if t1 == n:
            break
        t += hop

def rms_energy(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))

# --------------------------
# Scoring helpers
# --------------------------
def score_window(clf, chunk: np.ndarray, candidate_labels: List[str]) -> Dict[str, float]:
    out = clf(chunk, candidate_labels=candidate_labels, hypothesis_template=HYPOTHESIS_TEMPLATE)
    return {d["label"]: float(d["score"]) for d in out}

def group_scores(label_scores: Dict[str, float], taxonomy: Dict[str, List[str]]) -> Dict[str, float]:
    return {flag: max((label_scores.get(p, 0.0) for p in phrases), default=0.0)
            for flag, phrases in taxonomy.items()}

def top_phrase_per_flag(label_scores: Dict[str, float], taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    out = {}
    for flag, phrases in taxonomy.items():
        if not phrases:
            out[flag] = ""
            continue
        best = max(phrases, key=lambda p: label_scores.get(p, 0.0))
        out[flag] = best
    return out

def make_flags_desc(gscores: Dict[str, float], top_phrase: Dict[str, str], threshold: float) -> Tuple[List[str], str]:
    fired = [f for f, s in gscores.items() if s >= threshold]
    fired_sorted = sorted(fired, key=lambda f: -gscores[f])
    bits = [f"{f} ({gscores[f]:.2f}, {top_phrase[f]})" for f in fired_sorted]
    desc = "; ".join(bits) if bits else "no distress cues"
    return fired_sorted, desc

# --------------------------
# Run
# --------------------------
def run(audio_path: str,
        bin_sec: float = 2.0,
        threshold: float = 0.35,
        energy_gate_db: float = -55.0,
        model_id: str = MODEL_ID,
        agitation_boost: float = 1.15,
        stress_boost: float = 1.05) -> pd.DataFrame:

    print(f"[INFO] Loading audio: {repr(_normalize_path(audio_path))}")

    clf = pipeline(task="zero-shot-audio-classification", model=model_id)
    wav, sr = load_audio(audio_path, target_sr=48000)

    # Candidate labels = all phrases + neutrals
    all_labels: List[str] = []
    for phrases in FLAG_TAXONOMY.values():
        all_labels.extend(phrases)
    all_labels.extend(NEUTRALS)

    rows = []
    hop = bin_sec

    for s0, s1, chunk in frame_iter(wav, sr, bin_sec, hop):
        t0 = s0 / sr
        t1 = s1 / sr

        # Energy gate (skip near-silence)
        rms = rms_energy(chunk)
        dbfs = 20.0 * np.log10(max(1e-9, rms))
        if dbfs < energy_gate_db:
            desc = "silence/low energy"
            row = {
                "file": os.path.basename(_normalize_path(audio_path)),
                "t0": round(t0, 2),
                "t1": round(t1, 2),
                "flags": "",
                "flags_detailed": "",
                "description": desc,
                "neutral_max": 0.0,
            }
            for f in FLAG_TAXONOMY.keys():
                row[f"scores_{f}"] = 0.0
                row[f"top_phrase_{f}"] = ""
            rows.append(row)

            # --- NEW: print the description for this window ---
            print(f"{t0:>6.2f}–{t1:>6.2f}s  {desc}")
            sys.stdout.flush()
            continue

        # Score window
        label_scores = score_window(clf, chunk, all_labels)
        neutral_max = max((label_scores.get(n, 0.0) for n in NEUTRALS), default=0.0)

        # Aggregate per flag + gentle boosts
        gs = group_scores(label_scores, FLAG_TAXONOMY)
        if "high_agitation" in gs and agitation_boost != 1.0:
            gs["high_agitation"] *= float(agitation_boost)
        if "stress" in gs and stress_boost != 1.0:
            gs["stress"] *= float(stress_boost)

        tp = top_phrase_per_flag(label_scores, FLAG_TAXONOMY)
        flags, desc = make_flags_desc(gs, tp, threshold)

        # Build row
        row = {
            "file": os.path.basename(_normalize_path(audio_path)),
            "t0": round(t0, 2),
            "t1": round(t1, 2),
            "flags": ";".join(flags),
            "description": desc,
            "neutral_max": round(neutral_max, 3),
        }

        # Detailed pairs [flag,reason] based on top contributing phrase
        details = []
        for f in flags:
            reason = canon_reason(tp.get(f, ""))
            details.append(f"[{f},{reason}]" if reason else f"[{f}]")
        row["flags_detailed"] = "".join(details)

        # Numeric debug columns
        for f in FLAG_TAXONOMY.keys():
            row[f"scores_{f}"] = round(gs.get(f, 0.0), 3)
            row[f"top_phrase_{f}"] = tp.get(f, "")

        rows.append(row)

        # --- NEW: print the description for this window ---
        print(f"{t0:>6.2f}–{t1:>6.2f}s  {desc}")
        sys.stdout.flush()

    return pd.DataFrame(rows)

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to input audio file")
    ap.add_argument("--out", default=None, help="Output CSV (default: streaming_CLAP.csv)")
    ap.add_argument("--bin-sec", type=float, default=2.0, help="Window size seconds")
    ap.add_argument("--threshold", type=float, default=None, help="Override threshold (0..1)")
    ap.add_argument("--sensitivity", choices=["high", "medium", "low"], default="high",
                    help="Sets a sensible threshold: high=0.1, medium=0.35, low=0.45")
    ap.add_argument("--energy-gate-db", type=float, default=-55.0, help="Skip bins below this RMS dBFS")
    ap.add_argument("--model", default=MODEL_ID, help="CLAP model id")
    ap.add_argument("--agitation-boost", type=float, default=1.15, help="Multiplier for high_agitation score")
    ap.add_argument("--stress-boost", type=float, default=1.05, help="Multiplier for stress score")
    args = ap.parse_args()

    norm_path = _normalize_path(args.audio)
    print(f"[DEBUG] incoming path: {repr(args.audio)}  -> normalized: {repr(norm_path)}")
    if not norm_path or not os.path.exists(norm_path):
        raise SystemExit(f"ERROR: Audio path invalid or not found: {repr(norm_path)}")

    thresh_by_sens = {"high": 0.1, "medium": 0.35, "low": 0.45}
    thr = args.threshold if args.threshold is not None else thresh_by_sens[args.sensitivity]

    df = run(
        norm_path,
        bin_sec=args.bin_sec,
        threshold=thr,
        energy_gate_db=args.energy_gate_db,
        model_id=args.model,
        agitation_boost=args.agitation_boost,
        stress_boost=args.stress_boost,
    )

    out_path = args.out or "streaming_CLAP.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
