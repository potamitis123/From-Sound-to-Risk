#!/usr/bin/env python3
"""
Generate 2-second diarization flags from a single audio file.

Outputs per window:
- overlap_pct, distinct_speakers, peak_concurrent, interruption_count,
  interruption_rate_per_min, speaker_switches
- flags: HIGH_OVERLAP;MANY_SPEAKERS;INTERRUPTION;RAPID_TURNS
- description: short summary

Install:
  pip install pyannote.audio>=3.1 torch torchaudio soundfile numpy pandas

HF access:
  Accept model cards, create token, pass via --hf-token or HUGGINGFACE_TOKEN.
"""

import warnings

# 1) torch_audiomentations deprecation notice (torchaudio backend)
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.set_audio_backend has been deprecated.*",
    category=UserWarning,
    module=r"torch_audiomentations\.utils\.io"
)

# 2) pyannote pooling "degrees of freedom <= 0" message
warnings.filterwarnings(
    "ignore",
    message=r".*degrees of freedom is <= 0.*",
    category=UserWarning,
    module=r"pyannote\.audio\.models\.blocks\.pooling"
)

import argparse
import os
import sys
from typing import List, Tuple, Dict, Set

import pandas as pd
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Annotation

# ---------- immediate, unbuffered printing ----------
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)  # py>=3.7
except Exception:
    pass

def say(s: str = ""):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()

# --------------------------
# Utility: load duration
# --------------------------
def audio_duration_sec(path: str) -> float:
    with sf.SoundFile(path) as f:
        return len(f) / float(f.samplerate)

# --------------------------
# Convert Annotation -> [(start, end, speaker)]
# --------------------------
def annotation_to_segments(ann: Annotation) -> List[Tuple[float, float, str]]:
    segs: List[Tuple[float, float, str]] = []
    for segment, _, label in ann.itertracks(yield_label=True):
        if segment.end > segment.start:
            segs.append((float(segment.start), float(segment.end), str(label)))
    segs.sort(key=lambda x: (x[0], x[1]))
    return segs

# --------------------------
# Compute window stats
# --------------------------
def compute_window_stats(
    segments: List[Tuple[float, float, str]],
    w0: float,
    w1: float,
    tie_start_first: bool = False,
) -> Dict[str, float]:
    """Compute overlap, counts, interruptions, switches in [w0, w1)."""

    events: List[Tuple[float, int, str]] = []  # (time, kind, spk); kind: -1=end, +1=start
    speakers_seen: Set[str] = set()

    for s0, s1, spk in segments:
        if s1 <= w0 or s0 >= w1:
            continue
        a = max(s0, w0)
        b = min(s1, w1)
        if b <= a:
            continue
        speakers_seen.add(spk)
        events.append((a, +1, spk))
        events.append((b, -1, spk))

    if not events:
        return {
            "overlap_time": 0.0,
            "overlap_pct": 0.0,
            "distinct_speakers": 0,
            "peak_concurrent": 0,
            "interruption_count": 0,
            "interruption_rate_per_min": 0.0,
            "speaker_switches": 0,
        }

    # Tie-breaking: default end-before-start (conservative). start-before-end is more lenient.
    if tie_start_first:
        events.sort(key=lambda e: (e[0], -e[1]))  # start(+1) before end(-1) at same t
    else:
        events.sort(key=lambda e: (e[0], e[1]))   # end(-1) before start(+1) at same t

    active: Set[str] = set()
    peak_concurrent = 0
    overlap_time = 0.0
    interruption_count = 0
    speaker_switches = 0
    last_single_speaker = None

    for i, (t, kind, spk) in enumerate(events):
        if i > 0:
            prev_t = events[i - 1][0]
            dt = t - prev_t
            if dt > 0:
                if len(active) >= 2:
                    overlap_time += dt
                if len(active) == 1:
                    only = next(iter(active))
                    if last_single_speaker is None:
                        last_single_speaker = only
                    elif only != last_single_speaker:
                        speaker_switches += 1
                        last_single_speaker = only
                else:
                    last_single_speaker = None

        if kind == -1:
            active.discard(spk)
        else:
            if len(active) >= 1 and spk not in active:
                interruption_count += 1
            active.add(spk)

        peak_concurrent = max(peak_concurrent, len(active))

    last_t = events[-1][0]
    if w1 > last_t:
        dt = w1 - last_t
        if len(active) >= 2:
            overlap_time += dt
        if len(active) == 1:
            only = next(iter(active))
            if last_single_speaker is None:
                last_single_speaker = only
            elif only != last_single_speaker:
                speaker_switches += 1

    win_len = max(1e-9, w1 - w0)
    overlap_pct = overlap_time / win_len
    inter_rate_per_min = interruption_count / win_len * 60.0

    return {
        "overlap_time": overlap_time,
        "overlap_pct": overlap_pct,
        "distinct_speakers": len(speakers_seen),
        "peak_concurrent": peak_concurrent,
        "interruption_count": interruption_count,
        "interruption_rate_per_min": inter_rate_per_min,
        "speaker_switches": speaker_switches,
    }

# --------------------------
# Flagging & description (configurable thresholds)
# --------------------------
def make_flags_desc(
    stats: Dict[str, float],
    overlap_thr: float,
    many_speakers_thr: int,
    interruption_thr: int,
    rapid_turns_thr: int,
) -> Tuple[List[str], str]:
    flags = []
    if stats["overlap_pct"] >= overlap_thr:
        flags.append("HIGH_OVERLAP")
    if stats["peak_concurrent"] >= many_speakers_thr:
        flags.append("MANY_SPEAKERS")
    if stats["interruption_count"] >= interruption_thr:
        flags.append("INTERRUPTION")
    if stats["speaker_switches"] >= rapid_turns_thr:
        flags.append("RAPID_TURNS")

    bits = []
    if stats["overlap_pct"] > 0:
        bits.append(f"overlap {stats['overlap_pct']*100:.0f}%")
    bits.append(f"{stats['distinct_speakers']} spk")
    if stats["peak_concurrent"] > 1:
        bits.append(f"peak {int(stats['peak_concurrent'])}x")
    if stats["interruption_count"] > 0:
        bits.append(f"{int(stats['interruption_count'])} intr")
    if stats["speaker_switches"] > 0:
        bits.append(f"{int(stats['speaker_switches'])} switches")
    if not bits:
        bits.append("no activity")

    return flags, ", ".join(bits)

# --------------------------
# Run
# --------------------------
def run(
    audio_path: str,
    hf_token: str,
    bin_sec: float = 2.0,
    use_gpu: bool = False,
    min_speakers: int = None,
    max_speakers: int = None,
    num_speakers: int = None,
    overlap_thr: float = 0.10,
    many_speakers_thr: int = 2,
    interruption_thr: int = 1,
    rapid_turns_thr: int = 1,
    tie_start_first: bool = False,
    quiet: bool = False,
):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if use_gpu:
        import torch
        if torch.cuda.is_available():
            pipeline.to("cuda")
        else:
            say("[warn] --use-gpu requested but CUDA not available; using CPU.")

    diarization = pipeline(
        audio_path,
        **({} if num_speakers is None else {"num_speakers": num_speakers}),
        **({} if min_speakers is None else {"min_speakers": min_speakers}),
        **({} if max_speakers is None else {"max_speakers": max_speakers}),
    )

    segments = annotation_to_segments(diarization)
    dur = audio_duration_sec(audio_path)

    rows = []
    t = 0.0
    while t < dur:
        w0 = t
        w1 = min(dur, t + bin_sec)
        stats = compute_window_stats(segments, w0, w1, tie_start_first=tie_start_first)
        flags, desc = make_flags_desc(
            stats,
            overlap_thr=overlap_thr,
            many_speakers_thr=many_speakers_thr,
            interruption_thr=interruption_thr,
            rapid_turns_thr=rapid_turns_thr,
        )

        # ---- NEW: immediate console print per chunk ----
        if not quiet:
            bracketed = "".join(f"[{f}]" for f in flags) if flags else ""
            say(f"{int(round(w0))}-{int(round(w1))} {bracketed}")

        rows.append({
            "file": os.path.basename(audio_path),
            "t0": round(w0, 2),
            "t1": round(w1, 2),
            "overlap_pct": round(stats["overlap_pct"], 3),
            "distinct_speakers": int(stats["distinct_speakers"]),
            "peak_concurrent": int(stats["peak_concurrent"]),
            "interruption_count": int(stats["interruption_count"]),
            "interruption_rate_per_min": round(stats["interruption_rate_per_min"], 2),
            "speaker_switches": int(stats["speaker_switches"]),
            "flags": ";".join(flags) if flags else "",
            "description": desc,
        })
        t += bin_sec

    return pd.DataFrame(rows)

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to audio file")
    ap.add_argument("--out", default=None, help="Output CSV (default: streaming_speakers.csv)")
    ap.add_argument("--bin-sec", type=float, default=2.0, help="Window size seconds")
    ap.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_TOKEN"), help="HF token")
    ap.add_argument("--use-gpu", action="store_true", help="Run on CUDA if available")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)

    # Sensitivity presets
    ap.add_argument("--preset", choices=["lenient", "default", "strict"], default="lenient",
                    help="Threshold preset (lenient=more flags, strict=fewer)")
    # Manual overrides
    ap.add_argument("--overlap-thr", type=float, default=None, help="HIGH_OVERLAP threshold (0..1) [default 0.10 lenient]")
    ap.add_argument("--many-speakers-thr", type=int, default=None, help="MANY_SPEAKERS threshold [default 2 lenient]")
    ap.add_argument("--interruption-thr", type=int, default=None, help="INTERRUPTION threshold [default 1]")
    ap.add_argument("--rapid-turns-thr", type=int, default=None, help="RAPID_TURNS threshold [default 1 lenient]")
    ap.add_argument("--tie-start-first", action="store_true",
                    help="Treat coincident start/end as overlap (boosts HIGH_OVERLAP slightly)")

    # NEW: silence printing if desired
    ap.add_argument("--quiet", action="store_true", help="Disable per-chunk console printing")

    args = ap.parse_args()

    if not args.hf_token:
        raise SystemExit(
            "Missing HF token. Provide via --hf-token or HUGGINGFACE_TOKEN env var.\n"
            "Accept model terms for pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1."
        )

    # Preset mapping
    presets = {
        "lenient": {"overlap_thr": 0.10, "many_speakers_thr": 2, "interruption_thr": 1, "rapid_turns_thr": 1},
        "default": {"overlap_thr": 0.20, "many_speakers_thr": 3, "interruption_thr": 1, "rapid_turns_thr": 2},
        "strict":  {"overlap_thr": 0.35, "many_speakers_thr": 4, "interruption_thr": 2, "rapid_turns_thr": 3},
    }
    thr = presets[args.preset].copy()

    # Apply manual overrides if provided
    if args.overlap_thr is not None:        thr["overlap_thr"] = args.overlap_thr
    if args.many_speakers_thr is not None:  thr["many_speakers_thr"] = args.many_speakers_thr
    if args.interruption_thr is not None:   thr["interruption_thr"] = args.interruption_thr
    if args.rapid_turns_thr is not None:    thr["rapid_turns_thr"] = args.rapid_turns_thr

    df = run(
        args.audio,
        hf_token=args.hf_token,
        bin_sec=args.bin_sec,
        use_gpu=args.use_gpu,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        num_speakers=args.num_speakers,
        overlap_thr=thr["overlap_thr"],
        many_speakers_thr=thr["many_speakers_thr"],
        interruption_thr=thr["interruption_thr"],
        rapid_turns_thr=thr["rapid_turns_thr"],
        tie_start_first=args.tie_start_first,
        quiet=args.quiet,
    )

    out_path = args.out or "streaming_speakers.csv"
    df.to_csv(out_path, index=False)
    say(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
