# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="ctranslate2"
)

import argparse
import unicodedata
import re
from pathlib import Path
import pandas as pd
import os
import sys

# --- STRONGER DLL FIX ---
def force_cuda_paths():
    base_path = sys.prefix 
    # Define the critical Nvidia folders
    paths_to_add = [
        os.path.join(base_path, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin'),
        os.path.join(base_path, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin')
    ]
    
    for p in paths_to_add:
        if os.path.exists(p):
            # 1. The modern fix
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            # 2. The "Old School" fix (Force it into System PATH)
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            print(f"[DLL Fix] Injected into PATH: {p}")
        else:
            print(f"[DLL Fix] WARNING: Folder not found: {p}")

try:
    force_cuda_paths()
except Exception as e:
    print(f"DLL Fix Failed: {e}")
# ------------------------
    
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

from faster_whisper import WhisperModel

# ----------------- helpers -----------------

def normalize_token(s: str) -> str:
    """Lowercase+strip, remove accents, letters+apostrophes only, then drop apostrophes."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z'?!]+", "", s)  # keep letters and punct we care about
    return s.replace("'", "")

# Punctuation policy: always keep ! and ? (even if zero-duration)
PUNCT_TO_ALWAYS_KEEP = {"!", "?"}
def has_keep_punct(s: str) -> bool:
    s = (s or "").strip()
    return any(ch in s for ch in PUNCT_TO_ALWAYS_KEEP)
def is_punct_only(s: str) -> bool:
    s = (s or "").strip()
    return s in {"!", "?", "?!", "!?", "…"}

# Filler phrases to suppress lexically (no probabilities needed)
FILLER_PATTERNS = [
    ["i", "dont", "know"],
    ["i", "dont", "know", "what", "im", "talking", "about"],
    ["i", "dont", "know", "what", "im", "talking", "about", "but", "i", "dont", "know"],
]
# Lexical vocabulary for a ratio-based filter
FILLER_VOCAB = {"i", "dont", "know", "what", "im", "talking", "about", "but"}

def find_filler_spans(words_norm):
    """Return index ranges that exactly match configured filler patterns."""
    n = len(words_norm); spans = []
    for pat in FILLER_PATTERNS:
        m = len(pat)
        if m == 0 or n < m:
            continue
        i = 0
        while i <= n - m:
            if all(words_norm[i + k] == pat[k] for k in range(m)):
                spans.append((i, i + m)); i += m
            else:
                i += 1
    return spans

def spans_to_index_set(spans):
    idxs = set()
    for i, j in spans:
        idxs.update(range(i, j))
    return idxs

def filler_lexical_ratio(words_norm):
    """Fraction of tokens that belong to filler vocabulary."""
    if not words_norm:
        return 0.0
    hits = sum(1 for w in words_norm if w in FILLER_VOCAB)
    return hits / float(len(words_norm))

# ----------------- main -----------------

def main():
    parser = argparse.ArgumentParser(
        description="Word-level timestamps to CSV (faster-whisper) with probability-free filler suppression."
    )
    parser.add_argument("file_path", type=str, help="Path to audio (wav/mp3/ogg/flac/m4a).")
    parser.add_argument("--lang", default="auto",
                        help="Language code (e.g., 'en','el','ru','uk') or 'auto' to auto-detect.")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model name ('tiny','base','small','medium','large-v3').")
    parser.add_argument("--out", default="streaming_ASR.csv", help="Output CSV filename.")

    # Decoder/VAD hygiene (helps with city noise)
    parser.add_argument("--vad", action="store_true", default=True,
                        help="Enable VAD filtering in faster-whisper. Default: on.")
    parser.add_argument("--no-speech-thr", type=float, default=0.92,
                        help="Segments with no_speech_prob >= thr are treated as non-speech. Default 0.92.")
    parser.add_argument("--logprob-min", type=float, default=-0.8,
                        help="Segments with avg_logprob < this are considered low-confidence; used only for extra suppression decisions.")
    parser.add_argument("--condition-prev", action="store_true", default=False,
                        help="condition_on_previous_text. Default: off (reduces repetition loops).")

    # Beam/temperature
    parser.add_argument("--beam-size", type=int, default=2,
                        help="Beam size (1=greedy). Greedy is fastest and reduces hallucinations in noisy audio.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. Keep 0.0 when using beam search; >0 only with beam-size=1 if you want stochastic decoding.")

    # Probability-free lexical suppression
    parser.add_argument("--drop-zero-dur", action="store_true", default=True,
                        help="Drop tokens with start==end unless they are ! or ?. Default: on.")
    parser.add_argument("--min-word-dur", type=float, default=0.0,
                        help="Drop words shorter than this many seconds (unless !/?). Default 0.0.")
    parser.add_argument("--suppress-fillers", action="store_true", default=True,
                        help="Suppress 'I don't know …' phrases without using probabilities. Default: on.")
    parser.add_argument("--filler-ratio-thr", type=float, default=0.7,
                        help="If >= this fraction of segment tokens are filler vocab, suppress those tokens. Default 0.7.")

    args = parser.parse_args()

    # Model
    model = WhisperModel(args.model, device=DEVICE)

    # Language selection
    lang_is_auto = args.lang.lower() in {"auto", "detect", "autodetect"}
    lang_param = None if lang_is_auto else args.lang

    # --- Transcribe with fallback to English if auto-detect fails ---
    def do_transcribe(force_lang=None):
        return model.transcribe(
            args.file_path,
            language=force_lang if force_lang is not None else lang_param,
            word_timestamps=True,
            vad_filter=args.vad,
            no_speech_threshold=args.no_speech_thr,
            log_prob_threshold=args.logprob_min,
            condition_on_previous_text=args.condition_prev,
            temperature=args.temperature if args.beam_size == 1 else 0.0,
            beam_size=args.beam_size,
            without_timestamps=False,
            suppress_blank=True,
        )

    try:
        segments, info = do_transcribe()
    except ValueError as e:
        # Typical when auto language detection returns no candidates: "max() arg is an empty sequence"
        if lang_is_auto:
            print("[WARN] Language auto-detection failed; defaulting to English ('en').")
            segments, info = do_transcribe(force_lang="en")
        else:
            raise

    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}")
    # info.language may be None in fallback edge cases; show args.lang as fallback display
    chosen_lang = getattr(info, "language", None) or (args.lang if not lang_is_auto else "en")
    print(f"Language chosen: {chosen_lang}")
    in_file = Path(args.file_path).name

    rows = []
    for segment in segments:
        if not segment.words:
            continue

        seg_words = []
        for w in segment.words:
            if w.start is None or w.end is None:
                continue
            raw = unicodedata.normalize("NFC", w.word or "")
            norm = normalize_token(raw)
            start = float(w.start); end = float(w.end)
            dur = max(0.0, end - start)
            zero = (dur <= 1e-6)
            seg_words.append({"raw": raw, "norm": norm, "start": start, "end": end, "dur": dur, "zero": zero})

        if not seg_words:
            continue

        words_norm = [d["norm"] for d in seg_words]

        # Probability-free filler detection
        suppress_idxs = set()
        if args.suppress_fillers:
            spans = find_filler_spans(words_norm)
            suppress_idxs |= spans_to_index_set(spans)
            if filler_lexical_ratio(words_norm) >= args.filler_ratio_thr:
                suppress_idxs |= set(range(len(seg_words)))

        # Emit survivors
        for idx, d in enumerate(seg_words):
            word = d["raw"]
            if not word.strip():
                continue

            keep_punct = has_keep_punct(word) or is_punct_only(word)

            if idx in suppress_idxs and not keep_punct:
                continue
            if args.drop_zero_dur and d["zero"] and not keep_punct:
                continue
            if d["dur"] < args.min_word_dur and not keep_punct:
                continue

            rows.append({"file": in_file, "start": d["start"], "end": d["end"], "class1": word})
            print(f"[{d['start']:.2f} -> {d['end']:.2f}] {word}")

    df = pd.DataFrame(rows, columns=["file", "start", "end", "class1"])
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
