# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import pandas as pd
import torch
import torchaudio

import models  # your local models module

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
SR = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_wav',
        type=Path,
        nargs='+',
        help='One or more 16kHz mono files (or multi-channel, will be averaged).'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        metavar=f"Public Checkpoint [{','.join(models.list_models())}] or Experiment Path",
        #nargs='?', default='SAT_T_2s'
        nargs='?', default='SAT_B_2s'
    )
    parser.add_argument(
        '-c', '--chunk_length',
        type=float,
        help='Chunk length (seconds) for inference',
        default=2.0
    )
    parser.add_argument(
        '--min-tail-sec',
        type=float,
        default=1.0,
        help='Skip the final partial chunk if its duration (sec) is below this threshold. Default: 1.0'
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('streaming_AST.csv'),
        help='Output CSV filename (default: streaming_AST.csv)'
    )
    args = parser.parse_args()

    # Class label index map (AudioSet display names)
    cl_lab_idxs_file = Path(__file__).parent / 'datasets/audioset/data/metadata/class_labels_indices.csv'
    label_maps = pd.read_csv(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
        if not cl_lab_idxs_file.exists() else cl_lab_idxs_file
    ).set_index('index')['display_name'].to_dict()

    # Load model
    model = getattr(models, args.model)(pretrained=True).to(DEVICE).eval()

    # We will store top-3 class NAMES only (no probabilities)
    rows = []  # dicts: {'file','start','end','class1','class2','class3'}

    with torch.no_grad():
        zero_cache = None
        if 'SAT' in args.model:
            # Initialize "silence" cache for streaming SAT variants
            *_unused, zero_cache = model(
                torch.zeros(1, int(model.cache_length / 100 * SR), device=DEVICE),
                return_cache=True
            )

        for wavpath in args.input_wav:
            wave, sr = torchaudio.load(str(wavpath))
            assert sr == SR, "Models are trained on 16kHz; please resample your input to 16kHz"
            # Mixdown to mono if needed
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            wave = wave.to(DEVICE, non_blocking=True).contiguous().float()

            # Iterate over chunks
            chunk_len_samples = int(args.chunk_length * sr)
            print(f"\n===== {str(wavpath)} (win={args.chunk_length:.2f}s) =====")
            for chunk_idx, chunk in enumerate(wave.split(chunk_len_samples, dim=-1)):
                # --- Skip tiny final partial chunk to avoid patch-embed crash ---
                actual_len = chunk.shape[-1]
                if actual_len < chunk_len_samples:
                    tail_sec = actual_len / sr
                    if tail_sec < args.min_tail_sec:
                        # Optional: uncomment to see what is skipped
                        # print(f"(skip tail {tail_sec:.2f}s < {args.min_tail_sec:.2f}s)")
                        break  # stop processing this file
                # ---------------------------------------------------------------

                chunk = chunk.to(DEVICE, non_blocking=True).contiguous().float()

                # Forward
                if zero_cache is not None:
                    output, zero_cache = model(chunk, cache=zero_cache, return_cache=True)
                    output = output.squeeze(0)  # [num_classes]
                else:
                    output = model(chunk).squeeze(0)

                # Top-3 labels (names only for CSV)
                probs3, labels3 = output.topk(3)  # tensors of shape [3]
                lab_idxs = [int(i) for i in labels3.cpu().tolist()]
                class_names = [label_maps.get(idx, str(idx)) for idx in lab_idxs]

                # For screen print: show top-1 with probability
                prob_top1 = round(float(probs3[0].item()), 3)
                label_top1 = class_names[0]

                # Time bounds for this (full) chunk
                start_t = float(chunk_idx * args.chunk_length)
                end_t = float((chunk_idx + 1) * args.chunk_length)

                # Print
                print(f"{start_t:.2f}-{end_t:.2f}s  {label_top1:<30} p={prob_top1:.3f}")

                # Accumulate row with filename + top-3 class names
                c1 = class_names[0] if len(class_names) > 0 else ""
                c2 = class_names[1] if len(class_names) > 1 else ""
                c3 = class_names[2] if len(class_names) > 2 else ""
                rows.append({
                    'file': Path(wavpath).name,   # input filename
                    'start': start_t,
                    'end': end_t,
                    'class1': c1,
                    'class2': c2,
                    'class3': c3
                })

    # Build DataFrame and save (file as first column)
    df = pd.DataFrame(rows, columns=['file', 'start', 'end', 'class1', 'class2', 'class3'])
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
