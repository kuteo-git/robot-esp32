# "Mai ơi" Wake-Word Training (Phase 1: model training only)

## Context

The Android client (`xiaozhi-android`) currently detects wake words via two vendored
engines: Snowboy (English "Alexa", trainable model but Kitt.AI's training service has
been shut down since 2020 — no way to train new `.umdl`/`.pmdl` models) and a black-box
prebuilt "OK Nabu" engine (`libmicro_wake_word_jni.so`, model compiled into the binary,
no source to rebuild). Neither can produce a custom Vietnamese wake word.

Goal: replace the wake word with **"Mai ơi"**, using
[microWakeWord](https://github.com/kahrendt/microWakeWord) — the actively maintained,
open-source, trainable project the existing "OK Nabu" engine is itself built from, so a
newly trained model will fit the same class of on-device engine already proven to work
on this hardware.

## Scope

This spec covers **Phase 1 only: producing a trained, evaluated `mai_oi.tflite` model**.
Getting that model running on the R1 device (building microWakeWord's C++ inference
library for Android, writing a new JNI wrapper, wiring it into `VoiceModule.kt` /
`Settings.kt` / `control.html` as a third wake engine) is deliberately out of scope —
it's a separate follow-up spec, to be written once this phase produces a model worth
integrating. There is no point designing the Android plumbing before confirming a
Vietnamese wake word trains to usable accuracy at all.

## Why local, and why this repo

The repo `robot-esp32` already runs **VieNeuTTS** locally and self-hosted
(`services/vieneu_server.py`, Python package `vieneu`, model weights cached under
`~/.cache/huggingface`, existing venv at `services/.venv`) for the assistant's own
Vietnamese speech output. It generates short Vietnamese utterances in roughly 0.5–1.2s
each with no per-call cost or rate limit, across 6 distinct preset voices (mixed
gender/region) plus arbitrary voice cloning via `encode_reference()`. This makes it a
ready-made source of synthetic training data for a Vietnamese wake word, so the training
pipeline lives here (`services/wakeword_training/`) rather than in a new location, and
rather than depending on cloud TTS or hand-recording hundreds of samples.

microWakeWord's training pipeline is CPU/small-GPU friendly (a small streaming
keyword-spotting CNN, not a large model), so training also runs locally on the Mac mini
— no Colab/cloud GPU dependency needed.

## Pipeline

```
[1] Environment setup  →  [2] Positive data (VieNeuTTS)  →  [3] Negative/background data
                                                                        ↓
[5] Evaluation  ←  [4] Train (upstream microWakeWord script, augmentation baked in)
```

All stages are file-based (WAV folders in / out, `.tflite` out of stage 4), so any stage
can be re-run independently without repeating earlier ones.

### 1. Environment setup

- New Python venv dedicated to this pipeline, **separate from `services/.venv`**:
  microWakeWord's training code pins specific TensorFlow/tflite-micro versions likely to
  conflict with `vieneu`'s dependencies.
- Clone `github.com/kahrendt/microWakeWord` into
  `services/wakeword_training/vendor/microWakeWord`, install its pinned requirements
  into the new venv.
- Stage 2 is the only stage that needs `vieneu` — it calls into `services/.venv`
  (subprocess or brief in-process import) just for audio generation, then hands plain
  WAV files to the training venv, which never needs `vieneu` installed.

### 2. Positive data generation — `generate_positives.py`

- Calls `Vieneu(...)` directly (as `vieneu_server.py` does), looping over all 6 built-in
  preset voices (Bình, Ly, Ngọc, Tuyên, Vĩnh, Đoan — spans gender and Bắc/Nam accent)
  crossed with a range of `temperature`/`top_k` values, plus post-hoc pitch-shift/speed
  variants (reusing the `_pitch_wav`/`_boost_wav` helpers already in
  `vieneu_server.py`) for additional variety without extra model calls.
- Target ~3,000–4,000 clean positive clips of "Mai ơi". Kept clean (not pre-augmented)
  because stage 4's training pipeline already applies background-noise mixing, room
  impulse response reverb, and SpecAugment at train time.
- Optional stretch (not required for v1): clone 1–2 real family voices via
  `encode_reference()` for extra diversity if real-world evaluation later shows a high
  false-reject rate on real voices.

### 3. Negative data generation

- **Standard background/negative datasets** microWakeWord's own documentation points to
  (generic noise + generic speech corpora) — downloaded as-is, used as a base layer.
  These are language-agnostic and mostly cover generic false-trigger risk.
- **Vietnamese hard negatives** — the part that matters most for this deployment, since
  the device sits in a Vietnamese-speaking household and the standard sets are mostly
  English speech. Using the same VieNeuTTS generation script as stage 2, synthesize:
  - a curated list of phrases phonetically close to "Mai ơi" (e.g. "Mài ơi", "Hai ơi",
    "Mai ới", "mai ơi anh ơi"), and
  - a batch of everyday Vietnamese sentences representative of normal household
    conversation.

### 4. Training

- Run entirely inside the dedicated training venv, using microWakeWord's own training
  code as-is (not reimplemented).
- Feed it the positive/negative WAV folders from stages 2–3 in the manifest format its
  training config expects; it handles spectrogram feature extraction and augmentation
  itself.
- Use upstream's default model architecture and augmentation settings for this first
  pass — no reason to deviate until evaluation shows a specific, concrete problem to fix.
- Runs on CPU/Metal on the Mac mini.
- Output: a single quantized `mai_oi.tflite`.

### 5. Evaluation

Two tiers, because synthetic-only evaluation would just grade the model against its own
data distribution:

- **Synthetic held-out set**: a slice of stage 2/3 data excluded from training (different
  voice/parameter combinations) — automated FRR/FAR sanity check immediately after
  training.
- **Real-world set**: genuine "Mai ơi" recordings captured through the Android app's
  *existing* mic test tool (`MicTest.kt`, `/api/mic/start`) on actual R1 hardware, plus
  real ambient/conversation recordings from the same tool as real-world negatives. This
  is the set that actually matters — it reflects the real mic, hardware AEC, and room
  acoustics the model will run against in production, which synthetic-only accuracy can
  mask.

**Success criteria** (starting point, to be tuned once baseline numbers exist — no prior
model to calibrate against): false-reject rate under ~5% on the real-world positive set
at the chosen detection threshold, and false accepts sufficiently rare across several
hours of real ambient/negative audio.

## Deliverable

`mai_oi.tflite` + a short evaluation report (FRR/FAR numbers, breakdown by voice/
condition, worst-performing cases). Android-side integration (JNI wrapper, native build,
wiring into `VoiceModule.kt`) is a separate follow-up spec, written only once this
model's numbers justify integrating it.

## Explicitly out of scope for Phase 1

- Any changes to the `xiaozhi-android` repo.
- Building microWakeWord's C++ inference library for Android or writing a JNI wrapper.
- Choosing/implementing the detection-threshold UI or control-panel wiring.
- Voice cloning beyond the optional stretch goal in stage 2.
