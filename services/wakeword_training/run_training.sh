#!/usr/bin/env bash
# Training invocation wrapper for the "Mai ơi" wake-word model.
#
# The real pipeline (resolved against vendor/microWakeWord's notebook +
# source -- see extract_features.py's and train.py's module docstrings for
# the full research trail and notebook-cell quotes):
#
#   Step A: raw WAV (data/positive, data/negative_vi/{hard,generic}, listed
#           in data/manifest.json by Task 8's prepare_manifest.py) ->
#           Ragged Mmap spectrogram features under data/features/, via
#           extract_features.py (mirrors notebook cells 5-7: Clips +
#           Augmentation + SpectrogramGeneration + RaggedMmap.from_generator).
#
#   Step B: build training_parameters.yaml combining data/features/ with
#           data/negative_standard/'s already-downloaded Ragged Mmap folders
#           (Task 7), then invoke the real training entry point --
#           `python -m microwakeword.model_train_eval` (notebook cell 10 is
#           itself already a `!python -m microwakeword.model_train_eval ...`
#           shell invocation, not notebook-kernel-only code) -- and copy the
#           resulting quantized streaming tflite to models/mai_oi.tflite.
#           All done by train.py.
#
# NOT run by this script: a full real training pass. `training_steps`
# defaults to upstream's own example (10000 steps), which is intentionally
# expensive; running it is a separate, manual pipeline step documented at
# the end of the project plan, once Tasks 5-7 have produced full-scale data.
set -euo pipefail
cd "$(dirname "$0")"
source .venv-train/bin/activate

# ---------------------------------------------------------------------------
# WARNING: augmentation is OFF by default in this invocation.
#
# No --rir-dir/--background-dir is passed to extract_features.py below, so
# AddBackgroundNoise and RIR augmentation are no-ops (empty impulse/background
# path lists -- see extract_features.py's module docstring and
# Augmentation's own identity-transform fallback when given an empty list).
# train.py's training_parameters.yaml also hardcodes
# time_mask_count/freq_mask_count to [0] (SpecAugment time/freq masking off),
# matching upstream's own notebook cell 9 example verbatim.
#
# This is fine for a toy/smoke-test run, but the spec assumes this
# augmentation is ACTIVE at real training time (that's why positive
# generation stays clean -- augmentation is deliberately deferred to here).
# Before training a production-quality model, supply real RIR/background
# datasets via extract_features.py's --rir-dir/--background-dir flags (edit
# the invocation below). This script does not source or wire up such
# datasets itself.
# ---------------------------------------------------------------------------
# Step A: feature extraction (raw WAV -> Ragged Mmap spectrogram features), WITH
# RIR + background-noise augmentation ON. mit_rirs = room impulse responses;
# fma_16k + audioset_16k = background noise mixed in at -5..+10 dB SNR (see
# extract_features.build_augmenter). This makes the negatives (chimes, robot
# voice, speech) sound like they do through the device's mic in a real room --
# the augmentation the Phase-1 model was trained WITHOUT.
python extract_features.py \
  --manifest data/manifest.json \
  --out-dir data/features \
  --rir-dir data/mit_rirs \
  --background-dir data/fma_16k \
  --background-dir data/audioset_16k

# Step B: build the training config (our features + data/negative_standard)
# and invoke the real training entry point, mixednet hyperparameters taken
# verbatim from the notebook's cell 10 example invocation.
python train.py \
  --features-dir data/features \
  --negative-standard-dir data/negative_standard \
  --train-dir models/mai_oi_train \
  --training-config training_parameters.yaml \
  --out models/mai_oi.tflite

echo "Trained model exported to models/mai_oi.tflite"
