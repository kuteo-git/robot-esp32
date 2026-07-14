# Mai ơi Wake Word Retrain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain `mai_oi.tflite` so real-world sounds — the robot's own chimes, its VieNeu voice, and Vietnamese speech/noise — no longer false-trigger the wake word, while real "Mai ơi" still wakes reliably.

**Architecture:** Reuse the existing `services/wakeword_training/` microWakeWord pipeline unchanged in structure; add three new negative-data sources (chimes, VieNeu voice, real Vietnamese speech), turn on augmentation, retrain, and evaluate against a realistic set instead of the current 4+4 toy eval. Inference code and the model I/O contract are untouched.

**Tech Stack:** Python 3.10 in `services/wakeword_training/.venv-train` (TensorFlow 2.21, microWakeWord, pymicro-features), VieNeu TTS on `http://127.0.0.1:8002`, existing scripts (`generate_negatives.py`, `tts_generate.py`, `audio_variants.py`, `extract_features.py`, `train.py`, `evaluate.py`).

## Global Constraints

- All Python runs inside `services/wakeword_training/.venv-train` (`source .venv-train/bin/activate`).
- Feature extraction MUST use the vendored pymicro-features frontend (`extract_features.py`) — never a reimplementation; a mismatch silently breaks detection.
- Model I/O contract is fixed: input `[1,3,40]` int8, output `[1,1]` uint8, stride-3 streaming. Only weights/quant-constants may change.
- Audio for training/eval: 16 kHz mono PCM16 WAV.
- New Python modules follow the repo convention: one script + a matching `tests/test_<name>.py`, run with `pytest`.
- **Ship gate (Task 9):** 0 false-accepts on all chime variants, real "Mai ơi" detection ≥ 90%, overall false-accepts < 0.5/hour on held-out negatives.
- Commits in this repo use author `teo <kuteo-git@users.noreply.github.com>`.

---

### Task 1: Verify training environment + baseline

**Files:**
- Modify: none (verification only)
- Reference: `services/wakeword_training/run_training.sh`, `.venv-train/`

**Interfaces:**
- Produces: a confirmed-working env; the current `models/mai_oi.tflite` re-evaluated as the baseline to beat.

- [ ] **Step 1: Activate env and confirm core deps import**

Run:
```bash
cd services/wakeword_training && source .venv-train/bin/activate
python -c "import tensorflow, microwakeword, numpy, soundfile; print('deps OK')"
```
Expected: `deps OK` (TF 2.21.x).

- [ ] **Step 2: Confirm VieNeu TTS is reachable (needed for Task 3)**

Run: `curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8002/tts -H 'Content-Type: application/json' -d '{"input":"thử"}'`
Expected: `200`. If not, start it: `launchctl kickstart -k gui/501/com.user.robot-vieneu`.

- [ ] **Step 3: Run the existing test suite (baseline green)**

Run: `pytest -q`
Expected: all existing tests pass. If any fail, stop and fix env before proceeding.

- [ ] **Step 4: Commit a note capturing the baseline**

```bash
git add -A && git commit -m "chore(wakeword): verify retrain env + baseline green" --allow-empty
```

---

### Task 2: Chime hard-negative generator (fixes the deterministic loop)

**Files:**
- Create: `services/wakeword_training/gen_chime_negatives.py`
- Create: `services/wakeword_training/tests/test_gen_chime_negatives.py`
- Input: the three app chimes copied in Step 1 below.

**Interfaces:**
- Produces: `gen_chime_negatives(src_dir, out_dir, variants_per_clip) -> int` (count of WAVs written), writing 16 kHz mono PCM16 WAVs to `out_dir`. Consumed by the manifest step in Task 8.

- [ ] **Step 1: Copy the robot chimes into the training data**

Run:
```bash
mkdir -p data/negative_vi/chimes_src
cp ../../../xiaozhi-android/app/src/main/res/raw/end_of_request.wav \
   ../../../xiaozhi-android/app/src/main/res/raw/start_of_request.wav \
   ../../../xiaozhi-android/app/src/main/res/raw/listen_ding.wav \
   data/negative_vi/chimes_src/
```
(Adjust the relative path if the repos aren't siblings; the app repo is `xiaozhi-android`.)
Expected: 3 WAVs present in `data/negative_vi/chimes_src/`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_gen_chime_negatives.py
import soundfile as sf, numpy as np, os
from gen_chime_negatives import gen_chime_negatives

def test_generates_16k_mono_variants(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    sf.write(src / "chime.wav", (0.2*np.sin(np.linspace(0,50,8000))).astype("float32"), 16000)
    out = tmp_path / "out"
    n = gen_chime_negatives(str(src), str(out), variants_per_clip=5)
    assert n == 5
    files = list(out.glob("*.wav"))
    assert len(files) == 5
    data, sr = sf.read(files[0])
    assert sr == 16000 and data.ndim == 1
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `pytest tests/test_gen_chime_negatives.py -q`
Expected: FAIL (module `gen_chime_negatives` not found).

- [ ] **Step 4: Implement the generator**

```python
# gen_chime_negatives.py
"""Turn the robot's own chimes into augmented hard-negative WAVs so the wake
model learns to reject them (no-AEC device hears them through the mic)."""
import os, glob, random
import numpy as np, soundfile as sf
from audio_variants import to_16k_mono  # existing helper; resamples+mono

def _augment(x, rng):
    g = 10 ** (rng.uniform(-12, 3) / 20)          # volume -12..+3 dB
    x = x * g
    if rng.random() < 0.5:                         # simple reverb (decaying echoes)
        ir = np.zeros(int(16000 * rng.uniform(0.05, 0.3)), dtype="float32"); ir[0] = 1
        for _ in range(rng.randint(2, 6)):
            ir[rng.randint(1, len(ir)-1)] += rng.uniform(0.1, 0.5)
        x = np.convolve(x, ir)[: len(x)]
    if rng.random() < 0.5:                          # partial (start mid-chime)
        x = x[rng.randint(0, len(x)//2):]
    x = x + rng.normal(0, rng.uniform(0, 0.01), len(x)).astype("float32")  # mic hiss
    return np.clip(x, -1, 1).astype("float32")

def gen_chime_negatives(src_dir, out_dir, variants_per_clip=200, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    for src in sorted(glob.glob(os.path.join(src_dir, "*.wav"))):
        base = to_16k_mono(src)  # -> float32 mono 16k np.array
        stem = os.path.splitext(os.path.basename(src))[0]
        for i in range(variants_per_clip):
            sf.write(os.path.join(out_dir, f"{stem}_{i:04d}.wav"), _augment(base, rng), 16000)
            written += 1
    return written

if __name__ == "__main__":
    n = gen_chime_negatives("data/negative_vi/chimes_src", "data/negative_vi/chimes", 200)
    print(f"wrote {n} chime negatives")
```
If `audio_variants.to_16k_mono` has a different name, read `audio_variants.py` and use its actual resample-to-16k-mono helper (or inline `soundfile.read` + `librosa.resample` + mono mixdown).

- [ ] **Step 5: Run test + generate full set**

Run: `pytest tests/test_gen_chime_negatives.py -q && python gen_chime_negatives.py`
Expected: test PASS; `data/negative_vi/chimes/` gets 600 WAVs (3 chimes × 200).

- [ ] **Step 6: Commit**

```bash
git add gen_chime_negatives.py tests/test_gen_chime_negatives.py .gitignore
git commit -m "feat(wakeword): chime hard-negative generator"
```
(Ensure `data/` stays gitignored — check `.gitignore`.)

---

### Task 3: VieNeu robot-voice negatives (fixes own-voice false-wakes)

**Files:**
- Modify: `services/wakeword_training/tts_generate.py` (add a VieNeu HTTP backend)
- Create: `services/wakeword_training/gen_vieneu_negatives.py`
- Create: `services/wakeword_training/tests/test_gen_vieneu_negatives.py`

**Interfaces:**
- Consumes: VieNeu `/tts` (POST `{"input","voice"}` → WAV bytes).
- Produces: `gen_vieneu_negatives(sentences, voices, out_dir) -> int`, writing 16 kHz mono WAVs of the robot's voice saying non-wake sentences.

- [ ] **Step 1: Write the failing test (mock the HTTP call)**

```python
# tests/test_gen_vieneu_negatives.py
import soundfile as sf, numpy as np, io
from unittest import mock
from gen_vieneu_negatives import gen_vieneu_negatives

def _fake_wav_bytes():
    buf = io.BytesIO(); sf.write(buf, np.zeros(16000, "float32"), 16000, format="WAV"); return buf.getvalue()

def test_writes_one_wav_per_sentence_voice(tmp_path):
    with mock.patch("gen_vieneu_negatives._synth", return_value=_fake_wav_bytes()):
        n = gen_vieneu_negatives(["câu một", "câu hai"], ["Ngọc Lan"], str(tmp_path))
    assert n == 2
    assert len(list(tmp_path.glob("*.wav"))) == 2
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/test_gen_vieneu_negatives.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# gen_vieneu_negatives.py
"""Generate negatives in the robot's OWN VieNeu voice(s) so the model rejects
the robot's own TTS output (the 'wakes on its own voice' failure)."""
import os, requests, soundfile as sf, io, numpy as np
from audio_variants import to_16k_mono
VIENEU_URL = os.environ.get("VIENEU_URL", "http://127.0.0.1:8002/tts")

def _synth(text, voice):
    r = requests.post(VIENEU_URL, json={"input": text, "voice": voice}, timeout=60)
    r.raise_for_status(); return r.content

def gen_vieneu_negatives(sentences, voices, out_dir):
    os.makedirs(out_dir, exist_ok=True); n = 0
    for v in voices:
        for i, s in enumerate(sentences):
            wav = _synth(s, v)
            arr, sr = sf.read(io.BytesIO(wav))
            if arr.ndim > 1: arr = arr.mean(1)
            if sr != 16000: arr = to_16k_mono_arr(arr, sr)
            sf.write(os.path.join(out_dir, f"{v.replace(' ','_')}_{i:04d}.wav"), arr.astype("float32"), 16000)
            n += 1
    return n
```
Add `to_16k_mono_arr(arr, sr)` (librosa.resample) or reuse the helper from `audio_variants.py`. Build the sentence list from a large Vietnamese sentence corpus (reuse/extend `phrases.GENERIC_NEGATIVE_SENTENCES`, plus a few hundred sentences — pull from the Common Voice vi transcripts fetched in Task 4, or a bundled sentence file). Voices: the robot's presets (`Ngọc Lan`, `Ngọc Linh`, and the others from the active catalog).

- [ ] **Step 4: Run test + generate (a few hundred clips)**

Run: `pytest tests/test_gen_vieneu_negatives.py -q && python gen_vieneu_negatives.py`
Expected: test PASS; `data/negative_vi/robot_voice/` populated (aim ≥ 300 clips across voices).

- [ ] **Step 5: Commit**

```bash
git add gen_vieneu_negatives.py tests/test_gen_vieneu_negatives.py tts_generate.py
git commit -m "feat(wakeword): VieNeu robot-voice hard negatives"
```

---

### Task 4: Real Vietnamese speech negatives

**Files:**
- Modify: `services/wakeword_training/download_background_datasets.py`
- Modify: `services/wakeword_training/tests/test_download_background_datasets.py`

**Interfaces:**
- Produces: a `data/negative_standard/vi_speech/` (or `negative_vi/real_speech/`) folder of 16 kHz mono WAVs drawn from a real Vietnamese corpus (Mozilla Common Voice vi or Google FLEURS vi via `datasets`/`huggingface_hub`).

- [ ] **Step 1: Write a failing test for the new fetch/normalize function**

```python
def test_normalize_vi_clip_to_16k_mono(tmp_path):
    from download_background_datasets import normalize_to_16k_mono
    import soundfile as sf, numpy as np
    src = tmp_path/"a.wav"; sf.write(src, np.zeros((8000,2),"float32"), 8000)
    out = normalize_to_16k_mono(str(src), str(tmp_path/"o.wav"))
    data, sr = sf.read(out); assert sr == 16000 and data.ndim == 1
```

- [ ] **Step 2: Run to confirm fail; implement `normalize_to_16k_mono` + a `fetch_vi_speech(out_dir, max_clips)`**

Use `datasets.load_dataset("google/fleurs","vi_vn",split="test")` (small, no auth) or Common Voice vi; write `max_clips` normalized WAVs. Guard with a clear error if the dataset can't be fetched offline. Run: `pytest tests/test_download_background_datasets.py -q` → PASS.

- [ ] **Step 3: Fetch the corpus**

Run: `python download_background_datasets.py --vi-speech --max-clips 2000`
Expected: `data/negative_standard/vi_speech/` (or configured path) populated.

- [ ] **Step 4: Commit**

```bash
git add download_background_datasets.py tests/test_download_background_datasets.py
git commit -m "feat(wakeword): real Vietnamese speech negatives (FLEURS/Common Voice vi)"
```

---

### Task 5: Turn on augmentation + register new negatives in training config

**Files:**
- Modify: `services/wakeword_training/run_training.sh`
- Modify: `services/wakeword_training/training_parameters.yaml`
- Modify: `services/wakeword_training/prepare_manifest.py` (include the new chime + robot_voice dirs)

**Interfaces:**
- Produces: a training config that consumes the new negative feature dirs and runs with RIR + background-noise augmentation and SpecAugment masking ON.

- [ ] **Step 1: Ensure RIR + background datasets exist for augmentation**

Run: `python download_augmentation_data.py` (and `download_background_datasets.py` if RIRs come from there). Confirm non-empty RIR + background dirs.

- [ ] **Step 2: Edit `run_training.sh` to pass augmentation dirs**

Add `--rir-dir data/augmentation/rir --background-dir data/augmentation/background` (use the actual downloaded paths) to the `extract_features.py` invocation. Remove/silence the "augmentation OFF" warning once wired.

- [ ] **Step 3: Turn on SpecAugment masking + add new negative feature dirs**

In `training_parameters.yaml`: set `freq_mask_count`/`time_mask_count` to a small nonzero (e.g. `[2]`) with reasonable `*_max_size` (match upstream, e.g. `[5]`); add feature-dir entries for `data/features/negative_vi_chimes` (high `penalty_weight`, e.g. 4.0 — false-accepting a chime is the worst outcome), `data/features/negative_vi_robot_voice`, and `data/features/negative_standard/vi_speech`, each `truth: false` with sampling weights comparable to the existing negatives.

- [ ] **Step 4: Update `prepare_manifest.py`** to list the new `data/negative_vi/{chimes,robot_voice}` dirs so `extract_features.py` produces their feature folders. Run its test: `pytest tests/test_prepare_manifest.py -q` (update the test to cover the new dirs).

- [ ] **Step 5: Commit**

```bash
git add run_training.sh training_parameters.yaml prepare_manifest.py tests/test_prepare_manifest.py
git commit -m "feat(wakeword): enable augmentation + register chime/robot-voice/vi-speech negatives"
```

---

### Task 6: Realistic evaluation harness

**Files:**
- Modify: `services/wakeword_training/evaluate.py`
- Modify: `services/wakeword_training/tests/test_evaluate.py`

**Interfaces:**
- Produces: `evaluate_realistic(model, positives_dir, negatives_dirs, thresholds) -> dict` reporting, per threshold: positive detection rate, per-negative-source false-accept counts, and **false-accepts per hour** (using each negative's duration). Writes `reports/real_eval_t<th>.json`.

- [ ] **Step 1: Write the failing test** (feed a few synthetic positive/negative WAVs through a stub model; assert the dict has `detection_rate`, `false_accepts_per_hour`, and a per-source breakdown).
- [ ] **Step 2: Run to confirm fail.**
- [ ] **Step 3: Implement** the streaming-eval (reuse the existing frontend+interpreter path already in `evaluate.py`; add duration accounting → FA/hr; group negatives by source dir).
- [ ] **Step 4: Run test → PASS.** `pytest tests/test_evaluate.py -q`
- [ ] **Step 5: Commit** `feat(wakeword): realistic eval harness (detection rate + FA/hour by source)`.

---

### Task 7: Collect real "Mai ơi" recordings — **USER STEP**

**Files:**
- Create: `services/wakeword_training/data/real_eval/positive/*.wav` (gitignored)

- [ ] **Step 1:** Record ~20–50 clips of "Mai ơi": several speakers (you + family), varied distance (0.5–3 m), rooms, and volumes; include a few said right after the robot speaks. Phone voice-memo or the R1 mic is fine.
- [ ] **Step 2:** Convert to 16 kHz mono WAV and drop in `data/real_eval/positive/`:
```bash
for f in ~/mai_oi_recordings/*; do ffmpeg -i "$f" -ac 1 -ar 16000 "data/real_eval/positive/$(basename "${f%.*}").wav"; done
```
- [ ] **Step 3:** Also copy the raw chimes to `data/real_eval/neg_chimes/` and a held-out batch of VieNeu speech to `data/real_eval/neg_robot_voice/` for the eval negatives (the harness reads these).

---

### Task 8: Run the full pipeline (data → features → train → export)

**Files:**
- Produces: `services/wakeword_training/models/mai_oi.tflite` (retrained)

- [ ] **Step 1:** Regenerate all data (Tasks 2–4 outputs present), then run `bash run_training.sh`. Expected tail: `Trained model exported to models/mai_oi.tflite`. (Long-running; monitor for the eval metrics printed by `microwakeword.model_train_eval`.)
- [ ] **Step 2:** Confirm the model I/O contract unchanged:
```bash
python -c "import tensorflow as tf; i=tf.lite.Interpreter('models/mai_oi.tflite'); i.allocate_tensors(); d=i.get_input_details()[0]; o=i.get_output_details()[0]; print('in',d['shape'],d['dtype'],d['quantization']); print('out',o['shape'],o['dtype'],o['quantization'])"
```
Expected: input `[1 3 40] int8`, output `[1 1] uint8`. Record the printed quant scales/zero-points for Task 10.
- [ ] **Step 3:** Commit the model + reports: `feat(wakeword): retrained mai_oi.tflite on expanded negatives`.

---

### Task 9: Evaluate against the ship gate + pick threshold

**Files:**
- Produces: `reports/real_eval_t*.json`; a chosen operating threshold.

- [ ] **Step 1:** Run the harness across thresholds:
```bash
python evaluate.py --realistic --model models/mai_oi.tflite \
  --positives data/real_eval/positive \
  --negatives data/real_eval/neg_chimes data/real_eval/neg_robot_voice \
  --thresholds 0.3 0.5 0.7 0.9
```
- [ ] **Step 2:** Verify the **ship gate**: a threshold exists with **0 chime false-accepts**, real-"Mai ơi" detection ≥ 90%, and overall FA/hr < 0.5. Pick the threshold with the widest margin. If no threshold passes, iterate (more/weightier negatives, retrain) — do NOT ship.
- [ ] **Step 3:** Record the chosen threshold; commit the reports: `test(wakeword): realistic eval passes ship gate at t=<value>`.

---

### Task 10: Integrate the model into the app

**Files:**
- Modify: `xiaozhi-android/app/src/main/assets/mai_oi/mai_oi.tflite`
- Modify (if quant changed): `xiaozhi-android/app/src/main/java/info/dourok/voicebot/data/voice/MaiOiWakeWordDetector.kt` (the 4 quant constants), and the default `Settings.maiOiThreshold` to the Task 9 value.

- [ ] **Step 1:** Copy the retrained model: `cp services/wakeword_training/models/mai_oi.tflite ../xiaozhi-android/app/src/main/assets/mai_oi/mai_oi.tflite`.
- [ ] **Step 2:** Compare Task 8's printed quant constants to `inputScale`/`inputZeroPoint`/`outputScale`/`outputZeroPoint` in `MaiOiWakeWordDetector.kt`; update if different. Update the default threshold to the chosen operating point.
- [ ] **Step 3:** Commit in the app repo: `feat(wake): retrained Mai ơi model + threshold`.

---

### Task 11: On-device validation — **USER STEP**

- [ ] **Step 1:** Build & deploy the app to the R1 (existing build/deploy flow), select the "Mai ơi" engine in the control panel.
- [ ] **Step 2 (the decisive test):** Let a session end by 60s idle, and separately press the awake/end button → confirm the stop chime plays and it **stays asleep** (no re-wake loop).
- [ ] **Step 3:** Play TV/talk nearby for a few minutes → confirm no false wake. Say "Mai ơi" at 0.5–3 m several times → confirm reliable wake.
- [ ] **Step 4:** If all pass, done. If the chime still wakes it, capture `logcat -s MaiOiWakeWord` around the event (the score line) and feed back into Task 9 (add those exact clips as negatives, retrain).
