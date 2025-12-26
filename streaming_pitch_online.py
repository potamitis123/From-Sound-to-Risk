#!/usr/bin/env python3
"""
streaming_pitch_online.py — true streaming 2‑s pitch flags + optional [M]/[F] tag.

This refactor processes audio strictly in *online* mode:
- Reads the signal in fixed-size chunks (default 2 s), never loads the whole file.
- Runs torchcrepe per chunk (10 ms hop).
- Derives any needed normalization/statistics (e.g., robust z) ONLY from *previous* chunks
  via a rolling baseline (median/MAD) of voiced F0 values gathered so far.
- Emits one CSV row and one console line per processed chunk immediately.

Install:
  pip install soundfile torchcrepe numpy pandas librosa

Notes:
- Torch/torchcrepe will use CUDA automatically if available.
- If input sample rate != 16 kHz, each chunk is resampled on the fly.
- The last chunk may be shorter than --bin-sec (processed as-is).

Usage:
  python streaming_pitch_online.py input.wav --out streaming_pitch.csv
  python streaming_pitch_online.py input.wav --bin-sec 2 --z-high 1.0 --z-low -1.0 \
      --swing-hz 80 --rms-gate 0.05 --min-conf 0.2 \
      --min-voiced-ratio 0.20 --male-fmax 165 --female-fmin 180 --roll-sec 30
"""

import argparse, sys, os, csv
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torchcrepe
from collections import deque
from typing import Deque, Tuple


# -------------------- helpers --------------------

def segment_rms(y_seg: np.ndarray) -> float:
    """RMS over a waveform segment."""
    if y_seg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y_seg.astype(np.float32)))))


def pick_gender_label(fseg_hz: np.ndarray,
                      min_voiced_ratio: float,
                      male_fmax_hz: float,
                      female_fmin_hz: float) -> str:
    """
    Decide [M]/[F] using median F0 over voiced frames in THIS chunk.
    Requires: voiced_ratio >= min_voiced_ratio.
    """
    voiced_mask = np.isfinite(fseg_hz)
    if not np.any(voiced_mask):
        return ""
    voiced_ratio = float(np.mean(voiced_mask))
    if voiced_ratio < min_voiced_ratio:
        return ""
    f0_med = float(np.nanmedian(fseg_hz[voiced_mask]))
    if not np.isfinite(f0_med) or f0_med <= 0:
        return ""
    if f0_med <= male_fmax_hz:
        return "[M]"
    if f0_med >= female_fmin_hz:
        return "[F]"
    return ""  # ambiguous gap → no tag


def robust_z_from_baseline(x: np.ndarray, baseline_vals: np.ndarray) -> np.ndarray:
    """
    Compute robust z using median/MAD from a *baseline* vector (history).
    x: current-chunk vector (NaN ok)
    baseline_vals: only VOICED F0 from previous chunks (NaN already removed)
    Returns zeros if not enough history yet.
    """
    b = np.asarray(baseline_vals, float)
    if b.size < 8:  # need some history before z makes sense
        return np.zeros_like(x, dtype=float)
    med = np.nanmedian(b)
    mad = np.nanmedian(np.abs(b - med))
    scale = 1.4826 * mad if mad > 0 else (np.nanstd(b) if np.nanstd(b) > 0 else 1.0)
    return (x - med) / scale


def resample_to_16k(y: np.ndarray, sr_in: int, target_len: int) -> np.ndarray:
    """Resample a chunk to 16 kHz and pad/trim to target_len samples for consistent timing."""
    if sr_in == 16000:
        y16 = y.astype(np.float32, copy=False)
    else:
        y16 = librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=16000)
    # match requested length for stable bin timing (except possibly last bin)
    if target_len is not None:
        if len(y16) < target_len:
            pad = np.zeros(target_len - len(y16), dtype=np.float32)
            y16 = np.concatenate([y16, pad], axis=0)
        elif len(y16) > target_len:
            y16 = y16[:target_len]
    return y16.astype(np.float32, copy=False)


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to audio file")
    ap.add_argument("--out", default="streaming_pitch.csv", help="Output CSV")
    ap.add_argument("--bin-sec", type=float, default=2.0, help="Chunk size in seconds")

    # Pitch / loudness flags
    ap.add_argument("--z-high", type=float, default=1.0, help="median z(f0) > z_high → [HIGH_PITCH]")
    ap.add_argument("--z-low", type=float, default=-1.0, help="median z(f0) < z_low → [LOW_PITCH]")
    ap.add_argument("--swing-hz", type=float, default=80.0, help="std(f0) > swing_hz → [WIDE_PITCH_SWINGS]")
    ap.add_argument("--rms-gate", type=float, default=0.05, help="RMS > rms_gate → [LOUD]")

    # Torchcrepe / voicing gates
    ap.add_argument("--min-conf", type=float, default=0.2, help="torchcrepe periodicity min to trust f0 (0..1)")
    ap.add_argument("--min-voiced-ratio", type=float, default=0.20,
                    help="Min fraction of voiced frames per bin to emit [M]/[F]")

    # Gender thresholds (Hz)
    ap.add_argument("--male-fmax", type=float, default=165.0, help="Median F0 <= this → [M]")
    ap.add_argument("--female-fmin", type=float, default=180.0, help="Median F0 >= this → [F]")

    ap.add_argument("--model", choices=["tiny", "full"], default="tiny", help="torchcrepe model size")
    ap.add_argument("--roll-sec", type=float, default=30.0, help="History horizon for baseline (seconds)")
    ap.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA is available")
    args = ap.parse_args()

    # --- device & constants ---
    device = "cpu" if args.no_cuda else ("cuda:0" if torch.cuda.is_available() else "cpu")
    target_sr = 16000
    hop_length = int(target_sr / 100)  # 10 ms
    frames_per_sec = 100
    frames_per_bin = int(round(args.bin_sec * frames_per_sec))
    samples_per_bin_16k = int(round(args.bin_sec * target_sr))
    hist_max_frames = int(round(args.roll_sec * frames_per_sec))

    # Rolling voiced-F0 history (baseline for robust z) — ONLY previous frames
    history_voiced_f0: Deque[float] = deque(maxlen=hist_max_frames)

    # --- open output CSV ---
    out_abs = os.path.abspath(args.out)
    with open(out_abs, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["start", "end", "flags"])
        writer.writeheader()
        fcsv.flush()

        # --- streaming read ---
        with sf.SoundFile(args.audio) as f:
            sr_in = int(f.samplerate)
            t0 = 0.0

            while True:
                # compute input block size equal to ~bin-sec of input sr
                samples_per_bin_in = int(round(args.bin_sec * sr_in))
                y = f.read(samples_per_bin_in, dtype="float32", always_2d=False)
                if len(y) == 0:
                    break  # EOF

                # resample this chunk to 16 kHz for torchcrepe
                # for consistent bin timing, pad/trim to exact target length
                y16 = resample_to_16k(y, sr_in, target_len=samples_per_bin_16k)

                # Track wall-clock span for this output line
                # (use actual resampled length to be precise on the last bin)
                t1 = t0 + (len(y16) / target_sr)

                # --- torchcrepe per chunk ---
                audio = torch.tensor(y16, dtype=torch.float32, device=device)[None, :]  # (1, T)
                with torch.inference_mode():
                    f0, periodicity = torchcrepe.predict(
                        audio, target_sr, hop_length,
                        fmin=50.0, fmax=550.0,
                        model=args.model, device=device, batch_size=2048,
                        return_periodicity=True,
                        decoder=torchcrepe.decode.viterbi,
                    )
                f0 = f0.squeeze(0).detach().cpu().numpy()
                conf = periodicity.squeeze(0).detach().cpu().numpy()

                # mask low‑confidence frames
                f0_chunk = f0.astype(float, copy=True)
                f0_chunk[conf < args.min_conf] = np.nan

                # --- robust z using ONLY history from previous chunks ---
                baseline_vals = np.array(history_voiced_f0, dtype=float)  # voiced-only history
                z_chunk = robust_z_from_baseline(f0_chunk, baseline_vals)

                # --- flags for this chunk ---
                flags = []

                # HIGH/LOW pitch via median z over current chunk
                med_z = np.nanmedian(z_chunk) if np.any(np.isfinite(z_chunk)) else np.nan
                if np.isfinite(med_z):
                    if med_z > args.z_high:
                        flags.append("[HIGH_PITCH]")
                    elif med_z < args.z_low:
                        flags.append("[LOW_PITCH]")

                # WIDE swings via F0 std (absolute Hz) in this chunk
                if np.nanstd(f0_chunk) > args.swing_hz:
                    flags.append("[WIDE_PITCH_SWINGS]")

                # LOUD via simple RMS on waveform chunk
                if segment_rms(y16) > args.rms_gate:
                    flags.append("[LOUD]")

                # Optional gender tag (current chunk only; confidence/ratio gate)
                gender = pick_gender_label(
                    fseg_hz=f0_chunk,
                    min_voiced_ratio=args.min_voiced_ratio,
                    male_fmax_hz=args.male_fmax,
                    female_fmin_hz=args.female_fmin,
                )
                if gender:
                    flags.append(gender)

                # --- emit immediately ---
                line = "".join(flags)
                print(f"{int(round(t0))}-{int(round(t1))} {line}")
                writer.writerow({"start": round(t0, 2), "end": round(t1, 2), "flags": line})
                fcsv.flush()

                # --- update history AFTER emitting (no look‑ahead leakage) ---
                voiced_now = f0_chunk[np.isfinite(f0_chunk)]
                if voiced_now.size > 0:
                    # append elementwise keeping maxlen
                    free = history_voiced_f0.maxlen - len(history_voiced_f0)
                    if free <= 0:
                        # if full, extendleft/right costs; simplest: pop-left as needed then extend
                        to_add = voiced_now.tolist()
                        # keep last hist_max_frames voiced samples overall
                        # we add up to maxlen by trimming start
                        if len(to_add) >= history_voiced_f0.maxlen:
                            history_voiced_f0.clear()
                            history_voiced_f0.extend(to_add[-history_voiced_f0.maxlen:])
                        else:
                            # pop left to make space
                            for _ in range(len(to_add)):
                                if len(history_voiced_f0) >= history_voiced_f0.maxlen:
                                    history_voiced_f0.popleft()
                            history_voiced_f0.extend(to_add)
                    else:
                        history_voiced_f0.extend(voiced_now.tolist())

                # advance time
                t0 = t1

    print(f"[OK] wrote {out_abs}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)  # py>=3.7
    except Exception:
        pass
    main()
