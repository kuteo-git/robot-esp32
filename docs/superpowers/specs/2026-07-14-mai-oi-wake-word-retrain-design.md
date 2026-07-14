# "Mai ơi" Wake Word — Retrain for Real-World Robustness

## Context

The "Mai ơi" wake word (custom microWakeWord model, `services/wakeword_training/`,
deployed in the `xiaozhi-android` app as a selectable engine) false-wakes in real
use. The user's report:

- **Deterministic:** on session end (60s idle timeout **or** the awake/end
  button), the robot plays its stop chime and then **immediately re-wakes** — a
  self-sustaining loop. Reproduces every time.
- **Sporadic:** false-wakes on the robot's own voice, chimes, TV/background
  talk, and random noise.

By contrast the vendored "OK Nabu" engine is rock-solid. The difference is **not**
architecture or the inference pipeline (those work) — it is **model
discrimination**. OK Nabu was trained by Nabu Casa on large, varied data with
strong hard-negatives; "Mai ơi" was trained (Phase 1) on a narrow negative set
and validated on only **4 positives + 4 negatives** (`reports/real_hardware.json`).
That tiny eval looked clean (positives ~0.95–1.0, negatives ~0.00–0.02) but never
contained the sounds it actually fails on in the field.

## Root cause

The training negatives (`phrases.py`) are:
- 7 phonetic near-miss phrases ("Mài ơi", "Hai ơi", …) — good, keep.
- 10 generic Vietnamese sentences — too few, and **not** the acoustic sources
  that actually trigger false wakes.

**Absent from the negative set entirely:**
1. The robot's own **chimes** — `app/src/main/res/raw/end_of_request.wav`
   (the 2.3s session-end chime that drives the deterministic loop),
   `start_of_request.wav`, `listen_ding.wav`.
2. The robot's own **TTS voice** (VieNeu, e.g. Ngọc Lan) speaking arbitrary
   Vietnamese — the source of "wakes on its own voice".
3. Real, varied **Vietnamese speech** and **ambient/media/noise**.

Because the model never saw these, it happily scores them as "Mai ơi".

## Goal

Retrain `mai_oi.tflite` so that, on real R1 hardware:
- The stop/start/ding chimes **never** wake it (kills the deterministic loop).
- The robot's own VieNeu speech does not wake it.
- General Vietnamese speech, TV/media, and noise essentially never wake it.
- Real "Mai ơi" still wakes reliably.

This is a **data + evaluation** change to the existing Phase-1 pipeline. The
inference code (`MaiOiWakeWordDetector.kt`, `libmicro_features_jni`) and the model
I/O contract (`[1,3,40]` int8 → `[1,1]` uint8, stride-3 streaming) are unchanged;
only the `.tflite` weights and possibly its quant constants change. If the quant
scales/zero-points shift on re-export, update the four constants in
`MaiOiWakeWordDetector.kt` to match — no logic change.

## Approach: expand negatives, retrain, evaluate realistically

Reuse the entire existing pipeline (`generate_positives.py`, `generate_negatives.py`,
`audio_variants.py`, `extract_features.py`, `train.py`, `evaluate.py`, `.venv-train`).
The work is targeted additions to the **negative** data and a real evaluation set.

### 1. New hard-negative source — robot chimes (fixes the deterministic loop)

- Copy the three chimes from the app repo (`end_of_request.wav`,
  `start_of_request.wav`, `listen_ding.wav`) into the training data as
  **hard negatives**.
- Augment each heavily (`audio_variants.py`): room reverb / RIR, volume levels,
  partial/overlapping playback, mixed with background noise — to mimic how the
  device's own mic hears the chime through the room (no AEC). Target a few
  thousand augmented chime negatives so the class is well-represented.

### 2. New hard-negative source — the robot's own VieNeu voice

- Generate a large batch of VieNeu TTS clips (local, `:8002`) of varied Vietnamese
  sentences, **in the voice(s) the robot actually uses** (Ngọc Lan and the other
  preset voices), as negatives. This teaches the model to reject the robot's own
  speech specifically.
- Reuse `tts_generate.py`; add a VieNeu backend/voice list and a larger, more
  varied sentence corpus than the current 10 sentences.

### 3. Expand generic negatives — real Vietnamese speech + media/noise

- Add a real Vietnamese speech corpus (Mozilla **Common Voice vi** and/or
  **FLEURS vi**) as negatives via `download_background_datasets.py` — hundreds of
  distinct real speakers, the diversity synthetic sentences can't provide.
- Ensure background/ambient/media/music noise sets are downloaded and mixed
  (existing `download_background_datasets.py` / `download_augmentation_data.py`).

### 4. Positives — keep synthetic, add real for eval/fine-tune

- Keep the synthetic "Mai ơi" positive generation (multi-voice TTS + augmentation).
- Add ~20–50 **real** "Mai ơi" recordings (user + family, phone and/or the R1 mic,
  varied rooms/distances). Primary use: the **evaluation set**; optionally a light
  fine-tune. These are the ground truth for "does it actually wake".

### 5. Realistic evaluation (replaces the 4+4 eval)

Build an eval set that mirrors deployment, and report **operational** metrics, not
just accuracy on a toy set:
- **Positives:** the real "Mai ơi" recordings → report detection rate (target
  ≥ ~90%).
- **Negatives — must-not-fire:** every chime (raw + augmented), a held-out batch
  of VieNeu speech, held-out Common Voice vi, and noise/media → report
  **false-accepts per hour** (target: **0 on the chimes**, and < ~0.5/hr overall).
- Sweep the detection threshold and pick the operating point with the widest
  margin (the pipeline already produces `reports/synthetic_val_*_t*.json` per
  threshold; extend this to the real eval set).
- **Ship gate:** chimes = 0 false-accepts across all augmented variants, real
  "Mai ơi" detection ≥ target, overall FA/hr under target.

### 6. Export + integrate + on-device validation

- Re-export the streaming int8 `mai_oi.tflite`; print the input/output quant
  constants. If they differ from the current
  `MaiOiWakeWordDetector.kt` values, update the four constants there.
- Copy the model to `xiaozhi-android/app/src/main/assets/mai_oi/mai_oi.tflite`,
  rebuild, deploy to the R1.
- On-device smoke test (the decisive one): trigger a session end (idle 60s **and**
  the button) → confirm **no** re-wake; play TV/talk nearby → confirm no wake;
  say "Mai ơi" at varied distances → confirm reliable wake.

## Division of labor

- **Me (scripts + running the pipeline via shell on the Mac):** add the chime and
  VieNeu negative generators, wire in Common Voice/FLEURS vi, expand the negative
  sentence corpus, build the real-eval harness, run data-gen → feature extraction
  → training → evaluation → export, update the app asset + any quant constants.
- **User (hands-on):** record ~20–50 real "Mai ơi" clips (exact instructions
  provided); build & deploy the rebuilt Android app to the R1; run the final
  on-device smoke test.

## Risks

- **Vietnamese positive-voice diversity is limited** (few VN TTS voices). Mitigated
  by heavy augmentation, the phrase's distinctiveness, and the real recordings for
  eval/fine-tune. If detection generalization is weak, add more TTS engines (Edge
  TTS vi, cloud VN voices).
- **Over-suppression:** aggressively training against the VieNeu voice could make
  the model reject "Mai ơi" *spoken by a similar voice*. Mitigated by keeping
  positives voice-diverse and validating detection on the real recordings.
- **Local M4 training** (no CUDA): the model is tiny, so `tensorflow-metal`/CPU is
  adequate; the existing `.venv-train` + `run_training.sh` already target this host.

## Explicitly out of scope

- Changes to the inference architecture, `libmicro_features_jni`, or the
  `WakeWordDetector` interface (they work).
- Acoustic echo cancellation (AEC) on-device — a good model handles the no-AEC
  case, as OK Nabu proves.
- The full-duplex playback-gating mitigation (a separate, optional app-side change;
  not needed if the retrain succeeds).
- Retraining "OK Nabu" or touching the vendored engine.
