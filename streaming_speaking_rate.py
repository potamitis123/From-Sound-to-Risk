# Speaking-rate (de Jong & Wempe nuclei) with Parselmouth — 2s bins → flags
# pip install praat-parselmouth librosa numpy pandas

import argparse, numpy as np, pandas as pd, parselmouth, librosa, math

def intensity_track(sound, min_pitch_hz=50.0):
    # FIX: use time_step=None (auto) — time_step=0.0 is invalid in the Python API
    intensity = sound.to_intensity(minimum_pitch=min_pitch_hz, time_step=None, subtract_mean=True)
    t = intensity.xs()
    v = intensity.values[0]
    return np.asarray(t), np.asarray(v)

def pitch_track(sound, time_step=0.01, floor_hz=75.0, ceil_hz=450.0):
    return sound.to_pitch_ac(time_step=time_step, pitch_floor=floor_hz, pitch_ceiling=ceil_hz)

def local_maxima(x: np.ndarray) -> np.ndarray:
    if x.size < 3: return np.array([], dtype=int)
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1

def djw_syllable_nuclei(times, int_db, pitch, thr_db: float, mindip_db: float):
    peaks = local_maxima(int_db)
    peaks = peaks[int_db[peaks] > thr_db]
    if peaks.size == 0: return np.array([], dtype=float)

    valid_peak_times = []
    current_idx = None
    current_int = None
    for k in range(len(peaks) - 1):
        p = peaks[k]; q = peaks[k + 1]
        if current_idx is None:
            current_idx = p; current_int = int_db[p]
        dip = np.nanmin(int_db[current_idx:q+1])
        if abs(current_int - dip) > mindip_db:
            valid_peak_times.append(times[current_idx])
        current_idx = q; current_int = int_db[q]

    if not valid_peak_times: return np.array([], dtype=float)

    voiced = []
    for t in valid_peak_times:
        f0 = pitch.get_value_at_time(float(t))
        if f0 == f0 and f0 > 0:  # not NaN and positive
            voiced.append(t)
    return np.array(voiced, dtype=float)

def djw_threshold_from_intensity_db(int_db: np.ndarray, silence_db: float = -25.0) -> float:
    vals = int_db[np.isfinite(int_db)]
    if vals.size == 0: return -1000.0
    # Heuristic: take 99th percentile and offset by silence (negative) — tunable.
    max99 = np.quantile(vals, 0.99)
    thr = max99 + silence_db
    return max(thr, np.min(vals))

def per_bin_metrics(nuclei_times, bin_start, bin_end, times, int_db, thr_db):
    dur = max(1e-9, bin_end - bin_start)
    n_syll = int(np.sum((nuclei_times >= bin_start) & (nuclei_times < bin_end)))
    speech_rate = n_syll / dur

    mask = (times >= bin_start) & (times < bin_end)
    t_seg, v_seg = times[mask], int_db[mask]
    if t_seg.size >= 2:
        dt = float(np.median(np.diff(t_seg)))
        phon_time = float(np.sum(v_seg > thr_db) * dt)
    else:
        phon_time = 0.0
    articulation_rate = (n_syll / phon_time) if phon_time > 0 else 0.0
    phon_ratio = phon_time / dur

    return {
        "syllables": n_syll,
        "speech_rate": round(speech_rate, 3),
        "articulation_rate": round(articulation_rate, 3),
        "phonation_ratio": round(phon_ratio, 3),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", default="streaming_prosody.csv")
    ap.add_argument("--bin-sec", type=float, default=2.0)
    # DJW-style params
    ap.add_argument("--silence-db", type=float, default=-25.0, help="Intensity gate offset (dB)")
    ap.add_argument("--mindip-db", type=float, default=2.0, help="Minimum dip between peaks (dB)")
    ap.add_argument("--min-pitch-hz", type=float, default=50.0, help="Intensity minimum pitch (Hz)")
    ap.add_argument("--f0-floor", type=float, default=75.0)
    ap.add_argument("--f0-ceil", type=float, default=450.0)
    # Flag thresholds
    ap.add_argument("--rapid-thr", type=float, default=4.0, help="syll/s for [RAPID_RATE]")
    ap.add_argument("--slow-thr", type=float, default=2.0, help="syll/s for [SLOW_RATE]")
    ap.add_argument("--pause-heavy", dest="pause_heavy", type=float, default=0.5,
                    help="if phonation_ratio < this → [HEAVY_PAUSING]")
    args = ap.parse_args()

    y, sr = librosa.load(args.audio, sr=None, mono=True)
    snd = parselmouth.Sound(y, sr)

    times, int_db = intensity_track(snd, min_pitch_hz=args.min_pitch_hz)
    thr_db = djw_threshold_from_intensity_db(int_db, silence_db=args.silence_db)

    pitch = pitch_track(snd, time_step=0.01, floor_hz=args.f0_floor, ceil_hz=args.f0_ceil)
    nuclei = djw_syllable_nuclei(times, int_db, pitch, thr_db=thr_db, mindip_db=args.mindip_db)

    dur = float(len(y)) / sr
    rows = []
    t0 = 0.0
    while t0 < dur:
        t1 = min(dur, t0 + args.bin_sec)
        m = per_bin_metrics(nuclei, t0, t1, times, int_db, thr_db)
        flags = []
        if m["speech_rate"] >= args.rapid_thr: flags.append("[RAPID_RATE]")
        if 0 < m["speech_rate"] <= args.slow_thr: flags.append("[SLOW_RATE]")
        if m["phonation_ratio"] < args.pause_heavy: flags.append("[HEAVY_PAUSING]")
        rows.append({
            "start": round(t0, 2), "end": round(t1, 2),
            "flags": "".join(flags),
            "syllables": m["syllables"],
            "speech_rate": m["speech_rate"],
            "articulation_rate": m["articulation_rate"],
            "phonation_ratio": m["phonation_ratio"],
        })
        t0 = t1

    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out} (intensity_thr={thr_db:.1f} dB, mindip={args.mindip_db} dB)")

if __name__ == "__main__":
    main()
