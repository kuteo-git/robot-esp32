"""Prepare the user's real "Na Bi ơi" recordings for the keeper training run.

The raw recordings (~/Desktop/nabi_oi) are 16 kHz mono but hold the phrase inside
2-5 s of mostly silence, and about half are failed (near-silent) captures. The
microWakeWord augmenter aligns each clip's END near the window end, so a positive
must be trimmed down to just the phrase (leading/trailing silence removed), the
same shape as the synthetic TTS positives.

This script: drops near-silent clips, energy-trims the rest to the "Na Bi ơi"
phrase, then deterministically splits them into a held-out eval set (never
augmented into training, so the reported real-detection rate is honest) and a
training-source set. Output is 16 kHz mono PCM16.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import soundfile as sf

SR = 16000
FRAME = int(0.02 * SR)          # 20 ms energy frames
PAD_S = 0.15                    # keep a little air around the phrase
WINDOW_S = 1.5                  # phrase-length window we extract (matches synthetic positives)
MIN_PEAK = 0.03                 # below this the clip is a failed/silent capture
MIN_VOICED_S = 0.3              # need at least this much speech to be a real utterance


def _load(path: str) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
    return a


def _frame_rms(a: np.ndarray) -> np.ndarray:
    n = len(a) // FRAME
    return np.array([np.sqrt(np.mean(a[i * FRAME:(i + 1) * FRAME] ** 2)) for i in range(n)])


def trim_to_phrase(a: np.ndarray) -> np.ndarray | None:
    """Extract the single loudest WINDOW_S region -- the prominent utterance the user
    was recording -- so multi-attempt / robot-reply audio elsewhere in the clip can't
    be mistaken for the wake word by the end-aligning augmenter."""
    peak = float(np.abs(a).max())
    if peak < MIN_PEAK:
        return None
    fe = _frame_rms(a)
    thr = max(peak * 0.15, 0.01)
    if (fe > thr).sum() * 0.02 < MIN_VOICED_S:
        return None
    win = int(WINDOW_S / 0.02)  # window length in frames
    if len(fe) <= win:
        lo = 0
    else:
        csum = np.concatenate([[0.0], np.cumsum(fe)])
        energy = csum[win:] - csum[:-win]     # energy of every sliding window
        lo = int(np.argmax(energy))
    s = max(0, lo * FRAME - int(PAD_S * SR))
    e = min(len(a), (lo + win) * FRAME + int(PAD_S * SR))
    return a[s:e]


def main() -> None:
    src_glob = "/Users/lucnguyen/Desktop/nabi_oi/*.wav"
    eval_dir = "data/real_eval/positive_nabi_real"      # held out (honest eval)
    train_src = "data/real_eval/positive_nabi_train_src"  # fed to augmentation
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(train_src, exist_ok=True)

    kept = []
    for f in sorted(glob.glob(src_glob), key=lambda p: (len(p), p)):
        t = trim_to_phrase(_load(f))
        if t is None:
            print(f"  drop (silent/failed): {os.path.basename(f)}")
            continue
        kept.append((os.path.splitext(os.path.basename(f))[0].replace(" ", ""), t))

    # Deterministic split: every 4th clip is held out for eval, rest for training.
    n_eval = 0
    for i, (stem, t) in enumerate(kept):
        dest = eval_dir if i % 4 == 0 else train_src
        n_eval += dest == eval_dir
        sf.write(os.path.join(dest, f"{stem}.wav"), t.astype("float32"), SR, subtype="PCM_16")
        print(f"  keep {stem:12} dur={len(t)/SR:.2f}s -> {'EVAL' if dest==eval_dir else 'train'}")

    print(f"\nkept {len(kept)} voiced clips: {n_eval} held-out eval, {len(kept)-n_eval} training sources")


if __name__ == "__main__":
    main()
