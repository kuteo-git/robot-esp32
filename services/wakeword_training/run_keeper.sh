#!/bin/bash
# Keeper run for "Na Bi ơi": synthetic positives + the user's real recordings folded in.
# Detached (nohup) so it survives the assistant's session pausing.
cd /Users/lucnguyen/Documents/git/robot-esp32/services/wakeword_training || exit 1
source .venv-train/bin/activate
R=/tmp/nabi_keeper_results.txt

echo "=== keeper run started $(date) ===" >> "$R"

# Feature extraction (raw WAV -> Ragged Mmap), RIR + background + FAN noise ON.
rm -rf /Volumes/Data/mai-oi-training/features/* 2>/dev/null
python extract_features.py --manifest data/manifest.json --out-dir data/features \
  --rir-dir data/mit_rirs \
  --background-dir data/fma_16k --background-dir data/audioset_16k \
  --background-dir data/negative_vi/vi_speech \
  --background-dir data/real_eval/neg_env_train_src >> "$R" 2>&1 || { echo "EXTRACT FAILED" >> "$R"; exit 1; }

# Full 100k decayed run.
echo "--- train 100k started $(date) ---" >> "$R"
python train.py --features-dir data/features --negative-standard-dir data/negative_standard \
  --train-dir models/nabi_keeper_train --training-config training_parameters_full.yaml \
  --out models/mai_oi.tflite --training-steps 50000 30000 20000 >> "$R" 2>&1 \
  || { echo "TRAIN FAILED" >> "$R"; exit 1; }
cp -f models/mai_oi.tflite models/nabi_keeper.tflite

# Eval: honest REAL held-out (4 clips, never augmented) + synthetic held-out, vs all negatives.
NEG="--negatives data/real_eval/neg_nabi_hard --negatives data/real_eval/neg_env --negatives data/real_eval/neg_vi_speech --negatives data/real_eval/neg_chimes"
echo "--- EVAL real held-out $(date) ---" >> "$R"
python evaluate.py --realistic --model models/nabi_keeper.tflite \
  --positives data/real_eval/positive_nabi_real $NEG --thresholds 0.3 0.5 0.7 0.9 \
  --report-out reports/nabi_keeper_real.json 2>&1 | grep '^t=' >> "$R"
echo "--- EVAL synthetic held-out $(date) ---" >> "$R"
python evaluate.py --realistic --model models/nabi_keeper.tflite \
  --positives data/real_eval/positive_nabi $NEG --thresholds 0.5 0.7 0.9 \
  --report-out reports/nabi_keeper_synth.json 2>&1 | grep '^t=' >> "$R"

echo "=== ALL DONE $(date) ===" >> "$R"
touch /tmp/nabi_keeper.done
