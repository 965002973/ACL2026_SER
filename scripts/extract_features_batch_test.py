#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute frame length T (time steps) for Data2Vec/Wav2Vec-style ConvFeatureExtractionModel
using the exact formula used in your repo:
    L_out = floor((L_in - kernel_size) / stride + 1)

Feature encoder spec (from audio.py):
    [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]
i.e. kernels/strides:
    (10,5), (3,2)x4, (2,2), (2,2)
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ---- Conv spec from your repo ----
# format: (channels, kernel_size, stride)
FEATURE_ENCODER_SPEC: List[Tuple[int, int, int]] = (
    [(512, 10, 5)]
    + [(512, 3, 2)] * 4
    + [(512, 2, 2)]
    + [(512, 2, 2)]
)


@dataclass
class ConvStep:
    layer: int
    kernel: int
    stride: int
    Lin: int
    Lout: int


def compute_T_from_num_samples(num_samples: int,
                               spec: List[Tuple[int, int, int]] = FEATURE_ENCODER_SPEC
                               ) -> Tuple[int, List[ConvStep]]:
    """
    Returns:
      T: final time steps after the conv stack
      steps: per-layer (Lin -> Lout)
    """
    L = int(num_samples)
    steps: List[ConvStep] = []

    for i, (_c, k, s) in enumerate(spec, start=1):
        Lin = L
        # exact formula in your modules.py convert_padding_mask:
        # L_out = floor((L_in - k) / s + 1)
        L = int(np.floor((Lin - k) / s + 1))
        steps.append(ConvStep(layer=i, kernel=k, stride=s, Lin=Lin, Lout=L))

        if L <= 0:
            # Too-short audio can collapse to non-positive length
            break

    return L, steps


def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """
    Loads wav with soundfile. Returns mono float32 audio and sample rate.
    """
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)  # downmix to mono
    audio = audio.astype(np.float32, copy=False)
    return audio, int(sr)


def maybe_resample(audio: np.ndarray, sr: int, target_sr: Optional[int]) -> Tuple[np.ndarray, int]:
    if target_sr is None or target_sr == sr:
        return audio, sr
    # polyphase resample to reduce artifacts
    # up = target_sr/gcd, down = sr/gcd
    from math import gcd
    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    audio_rs = resample_poly(audio, up=up, down=down).astype(np.float32, copy=False)
    return audio_rs, target_sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="/data/zhaohaishu/Codes/AI-Synthesized-Voice-Generalization-main/training/test/1001_DFA_HAP_XX.wav", help="Path to a wav file")
    ap.add_argument("--target_sr", type=int, default=16000, help="Resample to this SR before computing T (default: 16000). Use 0 to disable.")
    ap.add_argument("--no_resample", default=False, action="store_true", help="Disable resampling; compute using original SR samples.")
    ap.add_argument("--show_layers", default=False, action="store_true", help="Print per-layer lengths.")
    args = ap.parse_args()

    audio, sr = load_wav_mono(args.wav)
    orig_n = len(audio)
    orig_dur = orig_n / sr if sr > 0 else float("nan")

    if args.no_resample or args.target_sr == 0:
        audio2, sr2 = audio, sr
    else:
        audio2, sr2 = maybe_resample(audio, sr, args.target_sr)

    n = len(audio2)
    dur = n / sr2 if sr2 > 0 else float("nan")

    T, steps = compute_T_from_num_samples(n)

    print(f"WAV: {args.wav}")
    print(f"Original: sr={sr}, samples={orig_n}, duration={orig_dur:.4f}s")
    if (sr2 != sr) or (n != orig_n):
        print(f"Used for T: sr={sr2}, samples={n}, duration={dur:.4f}s")
    else:
        print(f"Used for T: sr={sr2}, samples={n}, duration={dur:.4f}s (no resample)")

    print(f"Computed T = {T}")

    if args.show_layers:
        print("\nPer-layer lengths (L_in -> L_out):")
        for st in steps:
            print(f"  layer{st.layer}: k={st.kernel}, s={st.stride} | {st.Lin} -> {st.Lout}")


if __name__ == "__main__":
    main()
