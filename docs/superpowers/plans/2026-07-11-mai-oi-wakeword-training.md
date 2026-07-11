# "Mai ơi" Wake-Word Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling (data generation, negative sourcing, manifest prep, training invocation, evaluation) needed to produce and evaluate `mai_oi.tflite`, a microWakeWord model detecting the Vietnamese phrase "Mai ơi", using the local VieNeuTTS engine for synthetic training data.

**Architecture:** A standalone Python project at `services/wakeword_training/`, decoupled from the production `services/.venv` (except for the two generation scripts, which must run under `services/.venv` because that's where `vieneu` is installed). A dedicated venv (`.venv-train`) hosts everything else: microWakeWord (vendored from source), manifest prep, training invocation, and evaluation. Every stage is file-based (WAV folders → WAV folders → `.tflite`), so stages are independently re-runnable.

**Tech Stack:** Python 3.12, microWakeWord (vendored from `github.com/OHF-Voice/micro-wake-word`, the canonical upstream — `kahrendt/microWakeWord`, referenced in the design spec, is a fork of it), TensorFlow (via microWakeWord's own requirements), librosa (pitch/speed variation), soundfile, huggingface_hub, pytest.

## Global Constraints

- Positive phrase is exactly `"Mai ơi"` (spec: `docs/superpowers/specs/2026-07-11-mai-oi-wakeword-training-design.md`).
- Target ~3,000–4,000 clean positive clips (spec §2) — parameter grid must be sized to land in that range.
- Generation scripts (`generate_positives.py`, `generate_negatives.py`) must run under `services/.venv` (has `vieneu` installed); everything else runs under this project's own `.venv-train`.
- No pre-augmentation of generated audio — microWakeWord's training pipeline applies noise/RIR/SpecAugment itself (spec §2).
- Android-side integration is explicitly out of scope for this plan (spec, "Explicitly out of scope").
- All generated/vendored artifacts (`data/`, `models/`, `vendor/`, `.venv-train/`) must be gitignored — do not commit training data, model weights, or the vendored microWakeWord source.

---

## File Structure

```
services/wakeword_training/
├── .gitignore
├── setup_env.sh                       # Task 1
├── requirements.txt                   # Task 1
├── phrases.py                         # Task 2
├── audio_variants.py                  # Task 3
├── tts_generate.py                    # Task 4
├── generate_positives.py              # Task 5
├── generate_negatives.py              # Task 6
├── download_background_datasets.py    # Task 7
├── prepare_manifest.py                # Task 8
├── run_training.sh                    # Task 9
├── metrics.py                         # Task 10
├── evaluate.py                        # Task 11
└── tests/
    ├── test_phrases.py
    ├── test_audio_variants.py
    ├── test_tts_generate.py
    ├── test_generate_positives.py
    ├── test_generate_negatives.py
    ├── test_download_background_datasets.py
    ├── test_prepare_manifest.py
    └── test_metrics.py
    (evaluate.py's test lives at tests/test_evaluate.py, added in Task 11)
```

All automated tests in this plan run under `.venv-train` (created in Task 1) via
`.venv-train/bin/pytest`, including tests for the two generation scripts — their tests
always inject a fake TTS backend and never import `vieneu`, so they don't need
`services/.venv`. Only the real, full-scale *invocation* of `generate_positives.py` /
`generate_negatives.py` (a manual step after Task 11, not part of any task's test) needs
`services/.venv`.

---

### Task 1: Project scaffolding and dedicated training environment

**Files:**
- Create: `services/wakeword_training/.gitignore`
- Create: `services/wakeword_training/requirements.txt`
- Create: `services/wakeword_training/setup_env.sh`

**Interfaces:**
- Produces: a working venv at `services/wakeword_training/.venv-train` with `pytest`,
  `numpy`, `soundfile`, `librosa`, `huggingface_hub` installed, plus microWakeWord vendored
  at `services/wakeword_training/vendor/microWakeWord` with its own requirements installed
  into the same venv.

- [ ] **Step 1: Write `.gitignore`**

```
.venv-train/
vendor/
data/
models/
*.pyc
__pycache__/
```

- [ ] **Step 2: Write `requirements.txt`**

```
pytest==8.3.4
numpy>=1.26,<2.0
soundfile==0.12.1
librosa==0.10.2.post1
huggingface_hub==0.27.1
```

- [ ] **Step 3: Write `setup_env.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="/opt/homebrew/anaconda3/bin/python3.12"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Expected python3.12 at $PYTHON_BIN (matches services/.venv's base interpreter)." >&2
  echo "Adjust PYTHON_BIN in this script if your system differs." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip

if [ ! -d vendor/microWakeWord ]; then
  mkdir -p vendor
  # OHF-Voice/micro-wake-word is the canonical upstream (Home Assistant org);
  # kahrendt/microWakeWord (referenced in the design spec) is a fork of it.
  git clone https://github.com/OHF-Voice/micro-wake-word vendor/microWakeWord
fi
pip install -r vendor/microWakeWord/requirements.txt
pip install -r requirements.txt

echo "Done. Activate with: source services/wakeword_training/.venv-train/bin/activate"
```

- [ ] **Step 4: Run it and verify**

```bash
chmod +x services/wakeword_training/setup_env.sh
services/wakeword_training/setup_env.sh
services/wakeword_training/.venv-train/bin/python -c "import numpy, soundfile, librosa, huggingface_hub, pytest; print('ok')"
```

Expected: `ok` printed, no import errors. This will take several minutes (TensorFlow
install via microWakeWord's requirements).

- [ ] **Step 5: Commit**

```bash
cd services/wakeword_training
git add .gitignore requirements.txt setup_env.sh
git commit -m "wakeword_training: scaffold project and training venv setup"
```

---

### Task 2: Phrase lists

**Files:**
- Create: `services/wakeword_training/phrases.py`
- Test: `services/wakeword_training/tests/test_phrases.py`

**Interfaces:**
- Produces: `POSITIVE_PHRASE: str`, `HARD_NEGATIVE_PHRASES: list[str]`,
  `GENERIC_NEGATIVE_SENTENCES: list[str]` — consumed by Tasks 5 and 6.

- [ ] **Step 1: Write the failing test**

```python
# services/wakeword_training/tests/test_phrases.py
from phrases import POSITIVE_PHRASE, HARD_NEGATIVE_PHRASES, GENERIC_NEGATIVE_SENTENCES


def test_positive_phrase_is_mai_oi():
    assert POSITIVE_PHRASE == "Mai ơi"


def test_hard_negatives_nonempty_unique_and_distinct_from_positive():
    assert len(HARD_NEGATIVE_PHRASES) >= 5
    assert len(HARD_NEGATIVE_PHRASES) == len(set(HARD_NEGATIVE_PHRASES))
    assert POSITIVE_PHRASE not in HARD_NEGATIVE_PHRASES
    assert all(isinstance(p, str) and p.strip() for p in HARD_NEGATIVE_PHRASES)


def test_generic_negatives_nonempty_unique():
    assert len(GENERIC_NEGATIVE_SENTENCES) >= 5
    assert len(GENERIC_NEGATIVE_SENTENCES) == len(set(GENERIC_NEGATIVE_SENTENCES))
    assert all(isinstance(s, str) and s.strip() for s in GENERIC_NEGATIVE_SENTENCES)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/wakeword_training
.venv-train/bin/pytest tests/test_phrases.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'phrases'`.

- [ ] **Step 3: Write `phrases.py`**

```python
# services/wakeword_training/phrases.py
POSITIVE_PHRASE = "Mai ơi"

# Phonetically close to "Mai ơi" — trained as explicit negatives so the model
# doesn't false-trigger on near-miss Vietnamese speech.
HARD_NEGATIVE_PHRASES = [
    "Mài ơi",
    "Hai ơi",
    "Mai ới",
    "mai ơi anh ơi",
    "Nai ơi",
    "Mai đi",
    "Bài ơi",
]

# Everyday Vietnamese household speech, used as negatives so the model doesn't
# false-trigger during normal conversation.
GENERIC_NEGATIVE_SENTENCES = [
    "Hôm nay trời đẹp quá",
    "Con ăn cơm chưa",
    "Mẹ ơi con đói bụng",
    "Bao giờ mình đi chơi",
    "Anh đang làm gì đấy",
    "Chị lấy giúp em cái này với",
    "Bố đi làm về chưa",
    "Em muốn xem phim hoạt hình",
    "Tối nay ăn gì hả mẹ",
    "Con làm bài tập xong chưa",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-train/bin/pytest tests/test_phrases.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add phrases.py tests/test_phrases.py
git commit -m "wakeword_training: add positive/negative phrase lists"
```

---

### Task 3: Generation parameter grid

**Files:**
- Create: `services/wakeword_training/audio_variants.py`
- Test: `services/wakeword_training/tests/test_audio_variants.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PRESET_VOICES: list[str]`, `GenerationVariant` (frozen dataclass with fields
  `voice: str`, `temperature: float`, `top_k: int`, `pitch_semitones: int`,
  `speed_factor: float`, and property `tag: str`), and
  `build_variants(voices=PRESET_VOICES, temperatures=TEMPERATURES, top_ks=TOP_KS,
  pitch_semitones=PITCH_SEMITONES, speed_factors=SPEED_FACTORS) -> list[GenerationVariant]`.
  Consumed by Tasks 4, 5, 6.

**Confirmed `PRESET_VOICES` values:** read directly from
`services/.venv/lib/python3.12/site-packages/vieneu/assets/voices.json`'s `presets` keys
— they are ASCII, no diacritics: `Binh`, `Tuyen`, `Vinh`, `Doan`, `Ly`, `Ngoc`. Use these
exact strings (already reflected in Step 3 below).

- [ ] **Step 1: Write the failing test**

```python
# services/wakeword_training/tests/test_audio_variants.py
from audio_variants import GenerationVariant, build_variants, PRESET_VOICES


def test_preset_voices_nonempty():
    assert len(PRESET_VOICES) == 6
    assert len(set(PRESET_VOICES)) == 6


def test_variant_tag_is_unique_per_combo():
    variants = build_variants(
        voices=["A", "B"],
        temperatures=[0.7, 1.0],
        top_ks=[30],
        pitch_semitones=[0],
        speed_factors=[1.0],
    )
    assert len(variants) == 4  # 2 voices * 2 temperatures * 1 * 1 * 1
    tags = {v.tag for v in variants}
    assert len(tags) == 4


def test_build_variants_default_grid_size_is_in_target_range():
    # Spec targets ~3,000-4,000 clean positive clips for a single phrase;
    # this grid times 1 phrase should land in that range.
    variants = build_variants()
    assert 3000 <= len(variants) <= 4500


def test_variant_fields_roundtrip():
    v = GenerationVariant(voice="X", temperature=1.0, top_k=50, pitch_semitones=2, speed_factor=1.1)
    assert v.voice == "X"
    assert v.temperature == 1.0
    assert v.top_k == 50
    assert v.pitch_semitones == 2
    assert v.speed_factor == 1.1
    assert "X" in v.tag
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-train/bin/pytest tests/test_audio_variants.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'audio_variants'`.

- [ ] **Step 3: Write `audio_variants.py`**

```python
# services/wakeword_training/audio_variants.py
from dataclasses import dataclass

# Confirmed against vieneu/assets/voices.json's "presets" keys (ASCII, no diacritics).
PRESET_VOICES = ["Binh", "Doan", "Ly", "Ngoc", "Tuyen", "Vinh"]

TEMPERATURES = [0.7, 0.85, 1.0, 1.15, 1.3]
TOP_KS = [30, 50]
PITCH_SEMITONES = [-2, 0, 2]
SPEED_FACTORS = [0.9, 1.0, 1.1]


@dataclass(frozen=True)
class GenerationVariant:
    voice: str
    temperature: float
    top_k: int
    pitch_semitones: int
    speed_factor: float

    @property
    def tag(self) -> str:
        return (
            f"{self.voice}_t{self.temperature}_k{self.top_k}"
            f"_p{self.pitch_semitones}_s{self.speed_factor}"
        )


def build_variants(
    voices: list[str] = PRESET_VOICES,
    temperatures: list[float] = TEMPERATURES,
    top_ks: list[int] = TOP_KS,
    pitch_semitones: list[int] = PITCH_SEMITONES,
    speed_factors: list[float] = SPEED_FACTORS,
) -> list["GenerationVariant"]:
    return [
        GenerationVariant(voice, temperature, top_k, pitch, speed)
        for voice in voices
        for temperature in temperatures
        for top_k in top_ks
        for pitch in pitch_semitones
        for speed in speed_factors
    ]
```

Note: 6 voices × 5 temperatures × 2 top_ks × 3 pitches × 3 speeds = 540 variants, not
3,000+. Adjust the default grids (e.g. add more temperature steps, or add 2 more
speed/pitch steps) until `build_variants()` (all defaults) lands between 3,000 and 4,500
— pick whichever axis makes the most perceptual sense to expand (e.g. more temperature
samples for prosodic variety) and update this file and the test's expected range
together if you change the grid shape.

- [ ] **Step 4: Fix the grid size, then run test to verify it passes**

```bash
.venv-train/bin/pytest tests/test_audio_variants.py -v
```

Expected: PASS (4 tests), including the grid-size test.

- [ ] **Step 5: Commit**

```bash
git add audio_variants.py tests/test_audio_variants.py
git commit -m "wakeword_training: add TTS generation parameter grid"
```

---

### Task 4: TTS generation core

**Files:**
- Create: `services/wakeword_training/tts_generate.py`
- Test: `services/wakeword_training/tests/test_tts_generate.py`

**Interfaces:**
- Consumes: `GenerationVariant` from `audio_variants.py` (Task 3).
- Produces:
  - `TTSBackend` — a `Callable[[str, str, float, int], tuple[np.ndarray, int]]`
    (text, voice, temperature, top_k) → (mono float32 samples, sample_rate).
  - `make_vieneu_backend() -> TTSBackend` — real backend; imports `vieneu` lazily inside
    the function body so importing this module never requires `vieneu` to be installed.
  - `apply_pitch_speed(audio: np.ndarray, sample_rate: int, pitch_semitones: int, speed_factor: float) -> np.ndarray`.
  - `generate_dataset(texts: list[str], variants: list[GenerationVariant], backend: TTSBackend, out_dir: Path, label_prefix: str) -> list[Path]` —
    writes one 16-bit PCM WAV per (text, variant) pair into `out_dir`, returns the list
    of written paths. Consumed by Tasks 5 and 6.

- [ ] **Step 1: Write the failing test**

```python
# services/wakeword_training/tests/test_tts_generate.py
import numpy as np
import soundfile as sf

from audio_variants import GenerationVariant
from tts_generate import apply_pitch_speed, generate_dataset


def _fake_backend(text: str, voice: str, temperature: float, top_k: int):
    # 0.5s of silence at 16kHz, deterministic — no real TTS call.
    return np.zeros(8000, dtype=np.float32), 16000


def test_apply_pitch_speed_returns_nonempty_mono_array():
    audio = np.zeros(16000, dtype=np.float32)
    out = apply_pitch_speed(audio, sample_rate=16000, pitch_semitones=2, speed_factor=1.1)
    assert out.ndim == 1
    assert len(out) > 0


def test_generate_dataset_writes_one_wav_per_text_variant_pair(tmp_path):
    variants = [
        GenerationVariant("v1", 1.0, 50, 0, 1.0),
        GenerationVariant("v2", 1.0, 50, 0, 1.0),
    ]
    paths = generate_dataset(
        texts=["Mai ơi"],
        variants=variants,
        backend=_fake_backend,
        out_dir=tmp_path,
        label_prefix="pos",
    )
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.suffix == ".wav"
        data, sr = sf.read(str(p))
        assert sr == 16000
        assert len(data) > 0
        assert p.name.startswith("pos_")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-train/bin/pytest tests/test_tts_generate.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tts_generate'`.

- [ ] **Step 3: Write `tts_generate.py`**

```python
# services/wakeword_training/tts_generate.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from audio_variants import GenerationVariant

TTSBackend = Callable[[str, str, float, int], tuple[np.ndarray, int]]


def make_vieneu_backend() -> TTSBackend:
    """Real backend. Must be run under services/.venv (has `vieneu` installed)."""
    from vieneu import Vieneu

    tts = Vieneu(mode="standard", emotion="natural")

    def backend(text: str, voice: str, temperature: float, top_k: int) -> tuple[np.ndarray, int]:
        voice_data = tts.get_preset_voice(voice)
        audio, sample_rate = tts.infer(text, voice=voice_data, temperature=temperature, top_k=top_k)
        return np.asarray(audio, dtype=np.float32), sample_rate

    return backend


def apply_pitch_speed(
    audio: np.ndarray, sample_rate: int, pitch_semitones: int, speed_factor: float
) -> np.ndarray:
    out = audio
    if pitch_semitones != 0:
        out = librosa.effects.pitch_shift(out, sr=sample_rate, n_steps=pitch_semitones)
    if speed_factor != 1.0:
        out = librosa.effects.time_stretch(out, rate=speed_factor)
    return out.astype(np.float32)


def _clip_filename(label_prefix: str, text: str, variant: GenerationVariant) -> str:
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{label_prefix}_{variant.tag}_{text_hash}.wav"


def generate_dataset(
    texts: list[str],
    variants: list[GenerationVariant],
    backend: TTSBackend,
    out_dir: Path,
    label_prefix: str,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for text in texts:
        for variant in variants:
            audio, sample_rate = backend(text, variant.voice, variant.temperature, variant.top_k)
            audio = apply_pitch_speed(audio, sample_rate, variant.pitch_semitones, variant.speed_factor)
            path = out_dir / _clip_filename(label_prefix, text, variant)
            sf.write(str(path), audio, sample_rate, subtype="PCM_16")
            written.append(path)
    return written
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-train/bin/pytest tests/test_tts_generate.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tts_generate.py tests/test_tts_generate.py
git commit -m "wakeword_training: add TTS generation core with pitch/speed augmentation"
```

---

### Task 5: Positive dataset generation CLI

**Files:**
- Create: `services/wakeword_training/generate_positives.py`
- Test: `services/wakeword_training/tests/test_generate_positives.py`

**Interfaces:**
- Consumes: `POSITIVE_PHRASE` (Task 2), `build_variants` (Task 3), `generate_dataset`,
  `TTSBackend`, `make_vieneu_backend` (Task 4).
- Produces: `main(argv: list[str] | None = None, backend_factory: Callable[[], TTSBackend] = make_vieneu_backend) -> None`,
  writing WAVs to `--out-dir` (default `data/positive`). Real run (documented at the end
  of this plan) must use `services/.venv/bin/python`.

- [ ] **Step 1: Write the failing test**

```python
# services/wakeword_training/tests/test_generate_positives.py
import numpy as np

import generate_positives


def _fake_backend_factory():
    def backend(text, voice, temperature, top_k):
        return np.zeros(8000, dtype=np.float32), 16000
    return backend


def test_main_writes_one_file_per_variant(tmp_path, monkeypatch):
    # Shrink the grid so the test is fast and deterministic.
    import audio_variants

    monkeypatch.setattr(audio_variants, "PRESET_VOICES", ["v1", "v2"])
    generate_positives.main(
        argv=["--out-dir", str(tmp_path)],
        backend_factory=_fake_backend_factory,
    )
    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == len(audio_variants.build_variants())
    assert all(p.name.startswith("pos_") for p in wavs)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-train/bin/pytest tests/test_generate_positives.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'generate_positives'`.

- [ ] **Step 3: Write `generate_positives.py`**

```python
# services/wakeword_training/generate_positives.py
"""Generate synthetic "Mai ơi" positive clips.

Must be run with services/.venv/bin/python (has `vieneu` installed):
    services/.venv/bin/python services/wakeword_training/generate_positives.py \
        --out-dir services/wakeword_training/data/positive
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from audio_variants import build_variants
from phrases import POSITIVE_PHRASE
from tts_generate import TTSBackend, generate_dataset, make_vieneu_backend


def main(
    argv: list[str] | None = None,
    backend_factory: Callable[[], TTSBackend] = make_vieneu_backend,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/positive")
    args = parser.parse_args(argv)

    backend = backend_factory()
    variants = build_variants()
    written = generate_dataset(
        texts=[POSITIVE_PHRASE],
        variants=variants,
        backend=backend,
        out_dir=Path(args.out_dir),
        label_prefix="pos",
    )
    print(f"Wrote {len(written)} positive clips to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-train/bin/pytest tests/test_generate_positives.py -v
```

Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add generate_positives.py tests/test_generate_positives.py
git commit -m "wakeword_training: add positive dataset generation CLI"
```

---

### Task 6: Negative dataset generation CLI

**Files:**
- Create: `services/wakeword_training/generate_negatives.py`
- Test: `services/wakeword_training/tests/test_generate_negatives.py`

**Interfaces:**
- Consumes: `HARD_NEGATIVE_PHRASES`, `GENERIC_NEGATIVE_SENTENCES` (Task 2),
  `build_variants` (Task 3), `generate_dataset`, `TTSBackend`, `make_vieneu_backend` (Task 4).
- Produces: `main(argv=None, backend_factory=make_vieneu_backend) -> None`, writing hard
  negatives to `--out-dir`/`hard` and generic negatives to `--out-dir`/`generic`
  (default `--out-dir data/negative_vi`).

- [ ] **Step 1: Write the failing test**

```python
# services/wakeword_training/tests/test_generate_negatives.py
import numpy as np

import generate_negatives


def _fake_backend_factory():
    def backend(text, voice, temperature, top_k):
        return np.zeros(8000, dtype=np.float32), 16000
    return backend


def test_main_writes_hard_and_generic_subfolders(tmp_path, monkeypatch):
    import audio_variants

    monkeypatch.setattr(audio_variants, "PRESET_VOICES", ["v1"])
    generate_negatives.main(
        argv=["--out-dir", str(tmp_path)],
        backend_factory=_fake_backend_factory,
    )
    from phrases import HARD_NEGATIVE_PHRASES, GENERIC_NEGATIVE_SENTENCES
    variant_count = len(audio_variants.build_variants())

    hard_wavs = list((tmp_path / "hard").glob("*.wav"))
    generic_wavs = list((tmp_path / "generic").glob("*.wav"))
    assert len(hard_wavs) == len(HARD_NEGATIVE_PHRASES) * variant_count
    assert len(generic_wavs) == len(GENERIC_NEGATIVE_SENTENCES) * variant_count
    assert all(p.name.startswith("hardneg_") for p in hard_wavs)
    assert all(p.name.startswith("genneg_") for p in generic_wavs)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-train/bin/pytest tests/test_generate_negatives.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'generate_negatives'`.

- [ ] **Step 3: Write `generate_negatives.py`**

```python
# services/wakeword_training/generate_negatives.py
"""Generate synthetic Vietnamese negative clips (hard near-misses + generic speech).

Must be run with services/.venv/bin/python (has `vieneu` installed):
    services/.venv/bin/python services/wakeword_training/generate_negatives.py \
        --out-dir services/wakeword_training/data/negative_vi
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from audio_variants import build_variants
from phrases import GENERIC_NEGATIVE_SENTENCES, HARD_NEGATIVE_PHRASES
from tts_generate import TTSBackend, generate_dataset, make_vieneu_backend


def main(
    argv: list[str] | None = None,
    backend_factory: Callable[[], TTSBackend] = make_vieneu_backend,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/negative_vi")
    args = parser.parse_args(argv)

    backend = backend_factory()
    variants = build_variants()
    out_dir = Path(args.out_dir)

    hard = generate_dataset(
        texts=HARD_NEGATIVE_PHRASES,
        variants=variants,
        backend=backend,
        out_dir=out_dir / "hard",
        label_prefix="hardneg",
    )
    generic = generate_dataset(
        texts=GENERIC_NEGATIVE_SENTENCES,
        variants=variants,
        backend=backend,
        out_dir=out_dir / "generic",
        label_prefix="genneg",
    )
    print(f"Wrote {len(hard)} hard-negative clips and {len(generic)} generic-negative clips to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-train/bin/pytest tests/test_generate_negatives.py -v
```

Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add generate_negatives.py tests/test_generate_negatives.py
git commit -m "wakeword_training: add negative dataset generation CLI"
```

---

### Task 7: Standard background/negative dataset download

**Files:**
- Create: `services/wakeword_training/download_background_datasets.py`
- Test: `services/wakeword_training/tests/test_download_background_datasets.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `validate_dataset_dir(dir_path: Path) -> int` (returns count of valid audio
  files found, raises `ValueError` if zero), and a `main(argv=None) -> None` CLI that
  downloads upstream's standard negative/background datasets into
  `--out-dir` (default `data/negative_standard`) and calls `validate_dataset_dir` on the
  result.

**Before writing Step 3:** open `vendor/microWakeWord` (cloned in Task 1), specifically
`notebooks/basic_training_notebook.ipynb`, and find exactly which background-noise and
negative-speech datasets it downloads, and by what mechanism. Confirmed from upstream's
README: negative data is distributed as **pre-generated spectrogram features stored as
Ragged Mmap folders** (not raw WAV/audio files) at
`https://huggingface.co/datasets/kahrendt/microwakeword`, downloaded via
`huggingface_hub`. This means `AUDIO_SUFFIXES`/`validate_dataset_dir` below (written
assuming raw audio files) will likely need to change to validate Ragged Mmap folder
structure instead once you confirm the exact format from the notebook — adjust before
relying on this for real. Do not guess exact dataset repo IDs beyond the one confirmed
above — verify the precise repo path/subfolder the notebook actually uses.

- [ ] **Step 1: Write the failing test**

```python
# services/wakeword_training/tests/test_download_background_datasets.py
import pytest
import soundfile as sf
import numpy as np

from download_background_datasets import validate_dataset_dir


def test_validate_dataset_dir_counts_wav_files(tmp_path):
    for i in range(3):
        sf.write(str(tmp_path / f"clip_{i}.wav"), np.zeros(100, dtype=np.float32), 16000)
    (tmp_path / "readme.txt").write_text("not audio")

    count = validate_dataset_dir(tmp_path)
    assert count == 3


def test_validate_dataset_dir_raises_on_empty(tmp_path):
    with pytest.raises(ValueError):
        validate_dataset_dir(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-train/bin/pytest tests/test_download_background_datasets.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'download_background_datasets'`.

- [ ] **Step 3: Write `download_background_datasets.py`**

```python
# services/wakeword_training/download_background_datasets.py
"""Download microWakeWord's standard background-noise/negative-speech datasets.

Dataset identifiers below must match exactly what vendor/microWakeWord's own
training notebook/README uses — see the plan's note before this file's Step 3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg"}

# Fill in with the exact HF dataset repo IDs found in vendor/microWakeWord's
# training docs/notebook.
BACKGROUND_NOISE_REPO_ID = "REPLACE_WITH_UPSTREAM_REPO_ID"
NEGATIVE_SPEECH_REPO_ID = "REPLACE_WITH_UPSTREAM_REPO_ID"


def validate_dataset_dir(dir_path: Path) -> int:
    dir_path = Path(dir_path)
    count = sum(1 for p in dir_path.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if count == 0:
        raise ValueError(f"No audio files found under {dir_path}")
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/negative_standard")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    noise_dir = out_dir / "background_noise"
    speech_dir = out_dir / "negative_speech"
    snapshot_download(repo_id=BACKGROUND_NOISE_REPO_ID, repo_type="dataset", local_dir=str(noise_dir))
    snapshot_download(repo_id=NEGATIVE_SPEECH_REPO_ID, repo_type="dataset", local_dir=str(speech_dir))

    noise_count = validate_dataset_dir(noise_dir)
    speech_count = validate_dataset_dir(speech_dir)
    print(f"Downloaded {noise_count} background-noise files and {speech_count} negative-speech files to {out_dir}")


if __name__ == "__main__":
    main()
```

Replace `BACKGROUND_NOISE_REPO_ID` / `NEGATIVE_SPEECH_REPO_ID` with the real identifiers
found in Step 0 before running this for real. This is a real gap to close during
implementation, not a stylistic placeholder — the unit tests below only cover
`validate_dataset_dir` (pure function, no network), so they pass regardless; the actual
`main()` download only needs to work correctly when this task is run for real at the end
of the plan.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-train/bin/pytest tests/test_download_background_datasets.py -v
```

Expected: PASS (2 tests) — these test `validate_dataset_dir` only, no network calls.

- [ ] **Step 5: Commit**

```bash
git add download_background_datasets.py tests/test_download_background_datasets.py
git commit -m "wakeword_training: add standard negative/background dataset download script"
```

---

### Task 8: Manifest preparation (train/val split)

**Files:**
- Create: `services/wakeword_training/prepare_manifest.py`
- Test: `services/wakeword_training/tests/test_prepare_manifest.py`

**Interfaces:**
- Consumes: directories produced by Tasks 5, 6, 7 (`data/positive`,
  `data/negative_vi/hard`, `data/negative_vi/generic`, `data/negative_standard/**`).
- Produces: `build_manifest(positive_dir: Path, negative_dirs: list[Path], val_fraction: float = 0.15, seed: int = 0) -> dict`
  returning `{"train": {"positive": [str, ...], "negative": [str, ...]}, "val": {"positive": [...], "negative": [...]}}`
  (paths as strings, sorted, split deterministically via `seed`), and
  `write_manifest(manifest: dict, out_path: Path) -> None` (writes JSON). Consumed by
  Task 9's real training run.

**Before writing Step 3's final wiring in `main()`:** check `vendor/microWakeWord`'s
expected training config/manifest schema (its README or an example config yaml/json in
the repo) and adapt the JSON `write_manifest` produces so it's directly usable — or
produces a format Task 9's training invocation can trivially convert. Note the exact
adaptation needed as a comment in `prepare_manifest.py` once confirmed.

- [ ] **Step 1: Write the failing test**

```python
# services/wakeword_training/tests/test_prepare_manifest.py
import json

from prepare_manifest import build_manifest, write_manifest


def _make_wavs(dir_path, n):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (dir_path / f"clip_{i}.wav").write_bytes(b"")
    return dir_path


def test_build_manifest_splits_without_overlap_and_covers_all_files(tmp_path):
    pos_dir = _make_wavs(tmp_path / "positive", 20)
    neg_dir_a = _make_wavs(tmp_path / "neg_a", 10)
    neg_dir_b = _make_wavs(tmp_path / "neg_b", 10)

    manifest = build_manifest(pos_dir, [neg_dir_a, neg_dir_b], val_fraction=0.2, seed=0)

    train_pos = set(manifest["train"]["positive"])
    val_pos = set(manifest["val"]["positive"])
    train_neg = set(manifest["train"]["negative"])
    val_neg = set(manifest["val"]["negative"])

    assert len(train_pos) + len(val_pos) == 20
    assert train_pos.isdisjoint(val_pos)
    assert len(train_neg) + len(val_neg) == 20
    assert train_neg.isdisjoint(val_neg)
    assert len(val_pos) == 4  # 20 * 0.2
    assert len(val_neg) == 4  # 20 * 0.2


def test_build_manifest_is_deterministic_given_seed(tmp_path):
    pos_dir = _make_wavs(tmp_path / "positive", 20)
    neg_dir = _make_wavs(tmp_path / "neg", 20)

    m1 = build_manifest(pos_dir, [neg_dir], val_fraction=0.25, seed=42)
    m2 = build_manifest(pos_dir, [neg_dir], val_fraction=0.25, seed=42)
    assert m1 == m2


def test_write_manifest_writes_valid_json(tmp_path):
    pos_dir = _make_wavs(tmp_path / "positive", 4)
    neg_dir = _make_wavs(tmp_path / "neg", 4)
    manifest = build_manifest(pos_dir, [neg_dir], val_fraction=0.25, seed=0)

    out_path = tmp_path / "manifest.json"
    write_manifest(manifest, out_path)

    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == manifest
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-train/bin/pytest tests/test_prepare_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_manifest'`.

- [ ] **Step 3: Write `prepare_manifest.py`**

```python
# services/wakeword_training/prepare_manifest.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

AUDIO_SUFFIXES = {".wav"}


def _split(paths: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    paths = sorted(paths)
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    val_count = round(len(shuffled) * val_fraction)
    val = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])
    return train, val


def _collect_wavs(dir_path: Path) -> list[str]:
    return [str(p) for p in Path(dir_path).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES]


def build_manifest(
    positive_dir: Path, negative_dirs: list[Path], val_fraction: float = 0.15, seed: int = 0
) -> dict:
    positive_paths = _collect_wavs(positive_dir)
    negative_paths: list[str] = []
    for d in negative_dirs:
        negative_paths.extend(_collect_wavs(d))

    train_pos, val_pos = _split(positive_paths, val_fraction, seed)
    train_neg, val_neg = _split(negative_paths, val_fraction, seed)

    return {
        "train": {"positive": train_pos, "negative": train_neg},
        "val": {"positive": val_pos, "negative": val_neg},
    }


def write_manifest(manifest: dict, out_path: Path) -> None:
    Path(out_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-dir", default="data/positive")
    parser.add_argument(
        "--negative-dir",
        action="append",
        default=[
            "data/negative_vi/hard",
            "data/negative_vi/generic",
            "data/negative_standard/background_noise",
            "data/negative_standard/negative_speech",
        ],
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/manifest.json")
    args = parser.parse_args(argv)

    manifest = build_manifest(
        Path(args.positive_dir),
        [Path(d) for d in args.negative_dir],
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    write_manifest(manifest, Path(args.out))
    print(
        f"train: {len(manifest['train']['positive'])} positive / "
        f"{len(manifest['train']['negative'])} negative; "
        f"val: {len(manifest['val']['positive'])} positive / "
        f"{len(manifest['val']['negative'])} negative"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-train/bin/pytest tests/test_prepare_manifest.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add prepare_manifest.py tests/test_prepare_manifest.py
git commit -m "wakeword_training: add manifest builder with deterministic train/val split"
```

---

### Task 9: Training invocation wrapper

**Files:**
- Create: `services/wakeword_training/run_training.sh`

**Interfaces:**
- Consumes: `data/manifest.json` (Task 8 output), `vendor/microWakeWord` (Task 1).
- Produces: `models/mai_oi.tflite` when run for real (not part of this task's
  verification — see below).

**Before writing Step 1:** confirmed from upstream's README, training in
`OHF-Voice/micro-wake-word` runs through `notebooks/basic_training_notebook.ipynb`
(there is no plain `train.py` CLI), and it consumes **pre-generated spectrogram
features stored as Ragged Mmap folders**, not raw WAV files — so there is a feature-
extraction step missing between our WAV output (Tasks 5/6/7) and training that this
plan hasn't built yet. Read the notebook and `microwakeword/` package (both in the
cloned repo) to find: (a) the exact function/module that converts raw audio → Ragged
Mmap spectrogram features (likely something under `microwakeword/`), and (b) how the
notebook itself is meant to be run non-interactively (`jupyter nbconvert --execute`,
`papermill`, or a documented equivalent script). Update Step 1 below to run feature
extraction over `data/positive`, `data/negative_vi/**`, and
`data/negative_standard/**` first, writing Ragged Mmap folders, and only then invoke
training against those — not against `data/manifest.json`'s raw paths directly. This is
a real architecture gap surfaced by research during planning, not a stylistic
placeholder; resolve it against the actual cloned source before running Task 9 for real.

- [ ] **Step 1: Write `run_training.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv-train/bin/activate

# Step A: feature extraction (raw WAV -> Ragged Mmap spectrogram features).
# Fill in the real module/function found in vendor/microWakeWord's source per the
# note above — this is a placeholder shape only:
python -m microwakeword.feature_extraction \
  --positive-dir data/positive \
  --negative-dir data/negative_vi \
  --negative-dir data/negative_standard \
  --out-dir data/features

# Step B: train against extracted features, using notebooks/basic_training_notebook.ipynb
# as the reference for the real invocation (papermill or nbconvert --execute, or a
# script it delegates to) — fill in once confirmed:
jupyter nbconvert --to notebook --execute vendor/microWakeWord/notebooks/basic_training_notebook.ipynb \
  --output /dev/null

echo "Trained model expected at models/mai_oi.tflite — confirm output path matches the notebook's actual export step."
```

- [ ] **Step 2: Verify the invocation is well-formed (not a full training run)**

```bash
chmod +x services/wakeword_training/run_training.sh
services/wakeword_training/.venv-train/bin/python -c "import microwakeword; print(microwakeword.__file__)"
services/wakeword_training/.venv-train/bin/jupyter --version
```

Expected: both commands succeed, confirming the training package is importable and a
notebook runner is available before committing to the exact invocation shape in Step 1.
Update `run_training.sh` once the real feature-extraction and training entry points are
confirmed from the cloned source. A full training run is a manual step documented at the
end of this plan, not part of this task — it depends on real generated data from Tasks
5–7 and can take a long time.

- [ ] **Step 3: Commit**

```bash
cd services/wakeword_training
git add run_training.sh
git commit -m "wakeword_training: add training invocation wrapper"
```

---

### Task 10: Evaluation metrics

**Files:**
- Create: `services/wakeword_training/metrics.py`
- Test: `services/wakeword_training/tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EvalResult` (frozen dataclass: `false_reject_rate: float`,
  `false_accept_rate: float`, `num_positive: int`, `num_negative: int`,
  `num_false_rejects: int`, `num_false_accepts: int`) and
  `compute_metrics(positive_scores: list[float], negative_scores: list[float], threshold: float) -> EvalResult`.
  Consumed by Task 11.

- [ ] **Step 1: Write the failing test**

```python
# services/wakeword_training/tests/test_metrics.py
from metrics import compute_metrics


def test_all_correct():
    result = compute_metrics(positive_scores=[0.9, 0.8], negative_scores=[0.1, 0.2], threshold=0.5)
    assert result.false_reject_rate == 0.0
    assert result.false_accept_rate == 0.0
    assert result.num_false_rejects == 0
    assert result.num_false_accepts == 0


def test_all_wrong():
    result = compute_metrics(positive_scores=[0.1, 0.2], negative_scores=[0.9, 0.8], threshold=0.5)
    assert result.false_reject_rate == 1.0
    assert result.false_accept_rate == 1.0


def test_mixed():
    result = compute_metrics(positive_scores=[0.9, 0.1, 0.6], negative_scores=[0.1, 0.6, 0.2], threshold=0.5)
    assert result.num_positive == 3
    assert result.num_negative == 3
    assert result.num_false_rejects == 1  # the 0.1
    assert result.num_false_accepts == 1  # the 0.6
    assert result.false_reject_rate == 1 / 3
    assert result.false_accept_rate == 1 / 3


def test_empty_lists_do_not_divide_by_zero():
    result = compute_metrics(positive_scores=[], negative_scores=[], threshold=0.5)
    assert result.false_reject_rate == 0.0
    assert result.false_accept_rate == 0.0
    assert result.num_positive == 0
    assert result.num_negative == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-train/bin/pytest tests/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'metrics'`.

- [ ] **Step 3: Write `metrics.py`**

```python
# services/wakeword_training/metrics.py
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    false_reject_rate: float
    false_accept_rate: float
    num_positive: int
    num_negative: int
    num_false_rejects: int
    num_false_accepts: int


def compute_metrics(
    positive_scores: list[float], negative_scores: list[float], threshold: float
) -> EvalResult:
    num_false_rejects = sum(1 for s in positive_scores if s < threshold)
    num_false_accepts = sum(1 for s in negative_scores if s >= threshold)
    return EvalResult(
        false_reject_rate=(num_false_rejects / len(positive_scores)) if positive_scores else 0.0,
        false_accept_rate=(num_false_accepts / len(negative_scores)) if negative_scores else 0.0,
        num_positive=len(positive_scores),
        num_negative=len(negative_scores),
        num_false_rejects=num_false_rejects,
        num_false_accepts=num_false_accepts,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-train/bin/pytest tests/test_metrics.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_metrics.py
git commit -m "wakeword_training: add FRR/FAR evaluation metrics"
```

---

### Task 11: Evaluation CLI

**Files:**
- Create: `services/wakeword_training/evaluate.py`
- Test: `services/wakeword_training/tests/test_evaluate.py`

**Interfaces:**
- Consumes: `compute_metrics`, `EvalResult` (Task 10). Used against both the synthetic
  held-out set (Task 8's `val` split) and the real-world set (recorded via the Android
  app's existing mic test tool — see "Running the pipeline" below); same script, pointed
  at different directories.
- Produces: `TFLiteScorer` class wrapping `tensorflow.lite.Interpreter` with
  `score_wav_file(self, wav_path: Path) -> float`; `score_directory(scorer, dir_path: Path) -> list[float]`;
  `main(argv=None, scorer_factory=lambda model_path: TFLiteScorer(model_path)) -> None`
  writing a JSON report via `--report-out`.

**Before writing `TFLiteScorer`:** check `vendor/microWakeWord`'s own inference/streaming
example (it runs on-device the same way this evaluation needs to) for the exact input
tensor shape and any streaming-state handling the model expects — do not guess the
tensor shape.

- [ ] **Step 1: Write the failing test**

```python
# services/wakeword_training/tests/test_evaluate.py
import json

import numpy as np
import soundfile as sf

import evaluate


class _FakeScorer:
    def __init__(self, model_path):
        self.model_path = model_path

    def score_wav_file(self, wav_path):
        # Deterministic fake score based on filename so the test can assert exact values.
        return 0.9 if "pos" in str(wav_path) else 0.1


def test_score_directory_scores_every_wav(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.wav").write_bytes(b"")
    scores = evaluate.score_directory(_FakeScorer("unused"), tmp_path)
    assert len(scores) == 2


def test_main_writes_report_json(tmp_path):
    pos_dir = tmp_path / "positive"
    neg_dir = tmp_path / "negative"
    pos_dir.mkdir()
    neg_dir.mkdir()
    sf.write(str(pos_dir / "pos_1.wav"), np.zeros(100, dtype=np.float32), 16000)
    sf.write(str(neg_dir / "neg_1.wav"), np.zeros(100, dtype=np.float32), 16000)

    report_path = tmp_path / "report.json"
    evaluate.main(
        argv=[
            "--model", "unused.tflite",
            "--positive-dir", str(pos_dir),
            "--negative-dir", str(neg_dir),
            "--threshold", "0.5",
            "--report-out", str(report_path),
        ],
        scorer_factory=lambda model_path: _FakeScorer(model_path),
    )

    report = json.loads(report_path.read_text())
    assert report["num_positive"] == 1
    assert report["num_negative"] == 1
    assert report["false_reject_rate"] == 0.0
    assert report["false_accept_rate"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-train/bin/pytest tests/test_evaluate.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`.

- [ ] **Step 3: Write `evaluate.py`**

```python
# services/wakeword_training/evaluate.py
"""Score a trained mai_oi.tflite model against a positive/negative WAV directory pair.

Works for both the synthetic held-out set (Task 8's val split) and the real-world set
captured via the Android app's mic test tool — point --positive-dir/--negative-dir at
whichever set you want to evaluate.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Callable

from metrics import compute_metrics


class TFLiteScorer:
    def __init__(self, model_path: str):
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def score_wav_file(self, wav_path: Path) -> float:
        # NOTE(implementer): replace this body with the exact input tensor shape and
        # streaming-state handling confirmed against vendor/microWakeWord's own
        # inference example (see the note above Step 1 in the plan).
        raise NotImplementedError(
            "Fill in TFLiteScorer.score_wav_file per vendor/microWakeWord's inference contract"
        )


def score_directory(scorer, dir_path: Path) -> list[float]:
    return [scorer.score_wav_file(p) for p in sorted(Path(dir_path).glob("*.wav"))]


def main(
    argv: list[str] | None = None,
    scorer_factory: Callable[[str], object] = TFLiteScorer,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--positive-dir", required=True)
    parser.add_argument("--negative-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args(argv)

    scorer = scorer_factory(args.model)
    positive_scores = score_directory(scorer, Path(args.positive_dir))
    negative_scores = score_directory(scorer, Path(args.negative_dir))
    result = compute_metrics(positive_scores, negative_scores, args.threshold)

    Path(args.report_out).write_text(json.dumps(dataclasses.asdict(result), indent=2))
    print(
        f"FRR={result.false_reject_rate:.3f} "
        f"FAR={result.false_accept_rate:.3f} "
        f"({result.num_positive} positive, {result.num_negative} negative)"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-train/bin/pytest tests/test_evaluate.py -v
```

Expected: PASS (2 tests) — these exercise `score_directory`/`main`'s file-handling and
report-writing logic via `_FakeScorer`, not `TFLiteScorer` itself (which needs the real
tensor-shape fix noted above before it can run against an actual model).

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "wakeword_training: add evaluation CLI (FRR/FAR report)"
```

---

## Running the pipeline (manual, after all 11 tasks)

These are real, potentially long-running/resource-using invocations — not sized as
bite-sized tasks, and not run by an automated test:

1. `services/.venv/bin/python services/wakeword_training/generate_positives.py --out-dir services/wakeword_training/data/positive`
2. `services/.venv/bin/python services/wakeword_training/generate_negatives.py --out-dir services/wakeword_training/data/negative_vi`
3. Fill in the real dataset repo IDs in `download_background_datasets.py` (per Task 7's note), then: `services/wakeword_training/.venv-train/bin/python services/wakeword_training/download_background_datasets.py --out-dir services/wakeword_training/data/negative_standard`
4. `services/wakeword_training/.venv-train/bin/python services/wakeword_training/prepare_manifest.py --out services/wakeword_training/data/manifest.json`
5. `services/wakeword_training/run_training.sh` → produces `models/mai_oi.tflite`

   **Before running this step for real** (not just a toy/smoke-test model): note that
   `run_training.sh`'s default invocation of `extract_features.py` passes no
   `--rir-dir`/`--background-dir`, so `AddBackgroundNoise`/RIR augmentation are no-ops
   (empty impulse/background path lists — see `extract_features.py`'s module docstring
   and `Augmentation`'s own identity-transform fallback), and `train.py`'s
   `training_parameters.yaml` config hardcodes `time_mask_count`/`freq_mask_count` to
   `[0]` (SpecAugment masking off), matching upstream's own notebook cell 9 example
   verbatim. The spec assumes this augmentation is active at training time (that's *why*
   positive generation is allowed to stay clean — augmentation is deferred to here). The
   script itself now prints a loud warning about the RIR/background gap before running;
   read it. Running the default invocation trains a model without that augmentation —
   fine for a smoke test, but supply real `--rir-dir`/`--background-dir` datasets to
   `extract_features.py` (or edit `run_training.sh`'s invocation of it) before training a
   production-quality model. This fix does not source or wire up real RIR/background
   datasets — that's a separate, later step.
6. `services/wakeword_training/.venv-train/bin/python services/wakeword_training/evaluate.py --model models/mai_oi.tflite --manifest services/wakeword_training/data/manifest.json --split val --report-out reports/synthetic.json`
   (Task 8's `prepare_manifest.py` never materializes the val split as
   `data/positive_val`/`data/negative_val` directories — it writes individual file paths
   into `data/manifest.json`'s `"val"` key. `evaluate.py`'s `--manifest`/`--split` mode
   reads those paths directly instead of requiring a directory of pre-copied WAVs.)
7. Capture real-world "Mai ơi" recordings and real ambient negatives using the Android
   app's existing mic test tool (`MicTest.kt`, `http://<R1-ip>:8088` → mic test → record
   several takes of "Mai ơi" plus several minutes of normal household audio), save the
   WAVs into `services/wakeword_training/data/real_eval/positive` and
   `.../data/real_eval/negative`, then:
   `services/wakeword_training/.venv-train/bin/python services/wakeword_training/evaluate.py --model models/mai_oi.tflite --positive-dir data/real_eval/positive --negative-dir data/real_eval/negative --report-out reports/real.json`
8. Compare `reports/synthetic.json` and `reports/real.json` against the spec's success
   criteria (FRR under ~5% on the real-world set). If real-world FRR is much worse than
   synthetic FRR, that's the signal to revisit Task 4's voice-cloning stretch goal (real
   family voices via `encode_reference()`) before touching model architecture/hyperparameters.
