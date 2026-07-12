# VieNeu-TTS MLX Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in MLX backend to VieNeu-TTS's v3 Turbo engine and wire it into `robot-esp32/services/vieneu_server.py` as the default, with a one-env-var rollback to the existing pytorch/onnx engines, while bumping the underlying `vieneu` package from the ancient PyPI `3.0.5` to the current local checkout (content-equivalent to ~3.2.3, installed editable).

**Architecture:** A new `MLXV3TurboEngine` class in the VieNeu-TTS package (`src/vieneu/mlx/engine.py`) wraps the existing low-level MLX pipeline (`V3TurboPipelineMLX`) behind the same `.infer(phonemes=..., speaker_emb=..., ref_codes=..., style=..., ...) -> np.ndarray` contract that the package's ONNX/PyTorch engines already expose. `V3TurboVieNeuTTS.__init__` (`src/vieneu/v3turbo.py`) gets a new `backend="mlx"` branch that constructs it — no changes to that class's chunking/gap-joining/watermark/preset-voice logic, since the engine is a drop-in. `robot-esp32/services/vieneu_server.py` gets a `VIENEU_BACKEND` env var (default `mlx`) that selects which engine `Vieneu(mode="v3turbo", ...)` loads.

**Tech Stack:** Python 3.12, MLX (`mlx`, `mlx-lm`), `transformers` (tokenizer/config only on the MLX path), `pytest` + `unittest.mock`, FastAPI/uvicorn (unchanged).

## Global Constraints

- Do not git-sync `~/Documents/git/VieNeu-TTS` to the `v3.2.3` tag — work from the current checkout (commit `5f758da`) plus its uncommitted `src/vieneu/mlx/` files, as-is. No `git fetch`/`merge`/`pull` in that repo.
- The MLX adapter lives in VieNeu-TTS's `src/vieneu/mlx/`, not duplicated into robot-esp32.
- MLX backend is preset-voices-only — no `add_voice`/`encode_reference` support. Do not implement voice cloning for it.
- Default MLX quantization: `quantize_bits=4` (q4).
- `backend="auto"`'s existing onnx-on-CPU/pytorch-on-GPU resolution in `v3turbo.py` must be unchanged — `mlx` is only selected when explicitly requested (`backend="mlx"`), so other users of the published package see no behavior change.
- `MODE=standard` (v2 GGUF) in `vieneu_server.py` is untouched by this plan.
- Converted MLX weights already exist on disk at `/Volumes/Data/vieneu-mlx/v3turbo_backbone.safetensors` and `/Volumes/Data/vieneu-mlx/moss_decoder.safetensors` — do not reconvert them.
- Spec: `docs/superpowers/specs/2026-07-12-vieneu-mlx-backend-design.md`.

---

## Task 1: `MLXV3TurboEngine` in the VieNeu-TTS package

**Repo:** `~/Documents/git/VieNeu-TTS`

**Files:**
- Create: `src/vieneu/mlx/engine.py`
- Test: `tests/test_mlx_engine.py`

**Interfaces:**
- Produces: `vieneu.mlx.engine.MLXV3TurboEngine(backbone_repo: str, model_subfolder: Optional[str], moss_tokenizer: str, backbone_weights: Union[str, Path], moss_weights: Union[str, Path], quantize_bits: Optional[int] = 4, quantize_group_size: int = 64)` with methods `_resolve_style_id(style: Union[int, str]) -> int` and `infer(phonemes=None, text=None, ref_codes=None, speaker_emb=None, style="tu_nhien", use_ref_codes=True, temperature=0.8, top_k=25, top_p=0.95, max_new_frames=300, repetition_penalty=1.2, **kwargs) -> np.ndarray`. Consumed by Task 2.
- Consumes: `vieneu.mlx.v3turbo_pipeline.V3TurboPipelineMLX` (already exists — `__init__(backbone_weights, moss_weights, quantize_bits=None, quantize_group_size=64)`, `.generate_codes(phonemes, tokenizer, config, ref_codes, speaker_emb, style_token_id, temperature=0.8, top_k=25, top_p=0.95, repetition_penalty=1.2, max_new_frames=300, seed=None) -> np.ndarray` shape `(n_vq, T)`, `.synthesize(codes: np.ndarray) -> np.ndarray`); `vieneu._v3_turbo_engine.configuration_v3_turbo.VieNeuV3TurboConfig` (HF `PretrainedConfig` subclass, `.from_pretrained(repo, subfolder="")`, has `.style_labels: dict` and `.default_style_token_id: int`); `vieneu_utils.phonemize_text.phonemize_text_with_emotions(text: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mlx_engine.py`:

```python
"""Tests for vieneu.mlx.engine.MLXV3TurboEngine."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_missing_backbone_weights_raises(tmp_path):
    from vieneu.mlx.engine import MLXV3TurboEngine
    with pytest.raises(FileNotFoundError, match="backbone"):
        MLXV3TurboEngine(
            backbone_repo="dummy/repo", model_subfolder="update", moss_tokenizer="dummy/moss",
            backbone_weights=str(tmp_path / "missing_backbone.safetensors"),
            moss_weights=str(tmp_path / "missing_moss.safetensors"),
        )


def test_missing_moss_weights_raises(tmp_path):
    from vieneu.mlx.engine import MLXV3TurboEngine
    backbone = tmp_path / "backbone.safetensors"
    backbone.write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="MOSS"):
        MLXV3TurboEngine(
            backbone_repo="dummy/repo", model_subfolder="update", moss_tokenizer="dummy/moss",
            backbone_weights=str(backbone),
            moss_weights=str(tmp_path / "missing_moss.safetensors"),
        )


@pytest.fixture
def engine(tmp_path):
    from vieneu.mlx.engine import MLXV3TurboEngine
    backbone = tmp_path / "backbone.safetensors"
    moss = tmp_path / "moss.safetensors"
    backbone.write_bytes(b"fake")
    moss.write_bytes(b"fake")
    fake_config = MagicMock()
    fake_config.style_labels = {"tu_nhien": 16, "tin_tuc": 17}
    fake_config.default_style_token_id = 16
    with patch("vieneu.mlx.engine.AutoTokenizer") as mock_tok_cls, \
         patch("vieneu.mlx.engine.VieNeuV3TurboConfig") as mock_cfg_cls, \
         patch("vieneu.mlx.engine.V3TurboPipelineMLX") as mock_pipeline_cls:
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_cfg_cls.from_pretrained.return_value = fake_config
        mock_pipeline_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_instance
        eng = MLXV3TurboEngine(
            backbone_repo="dummy/repo", model_subfolder="update", moss_tokenizer="dummy/moss",
            backbone_weights=str(backbone), moss_weights=str(moss), quantize_bits=4,
        )
        eng._mock_pipeline = mock_pipeline_instance
        yield eng


def test_resolve_style_id_known_name(engine):
    assert engine._resolve_style_id("tin_tuc") == 17


def test_resolve_style_id_unknown_falls_back_to_default(engine):
    assert engine._resolve_style_id("nonexistent_style") == 16


def test_resolve_style_id_int_passthrough(engine):
    assert engine._resolve_style_id(5) == 5


def test_infer_calls_pipeline_generate_and_synthesize(engine):
    engine._mock_pipeline.generate_codes.return_value = np.zeros((16, 3), dtype=np.int64)
    engine._mock_pipeline.synthesize.return_value = np.ones(48000, dtype=np.float32)
    with patch("vieneu.mlx.engine.phonemize_text_with_emotions", return_value="ph on em es"):
        wav = engine.infer(text="Xin chào", style="tin_tuc", ref_codes=None, speaker_emb=None)
    assert isinstance(wav, np.ndarray)
    assert wav.shape == (48000,)
    engine._mock_pipeline.generate_codes.assert_called_once()
    call_kwargs = engine._mock_pipeline.generate_codes.call_args.kwargs
    assert call_kwargs["phonemes"] == "ph on em es"
    assert call_kwargs["style_token_id"] == 17
    engine._mock_pipeline.synthesize.assert_called_once()


def test_infer_empty_codes_returns_empty_array(engine):
    engine._mock_pipeline.generate_codes.return_value = np.zeros((16, 0), dtype=np.int64)
    with patch("vieneu.mlx.engine.phonemize_text_with_emotions", return_value="ph"):
        wav = engine.infer(text="", style="tu_nhien")
    assert isinstance(wav, np.ndarray)
    assert wav.shape == (0,)
    engine._mock_pipeline.synthesize.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Documents/git/VieNeu-TTS && .venv/bin/python -m pytest tests/test_mlx_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vieneu.mlx.engine'` (file doesn't exist yet).

- [ ] **Step 3: Write the implementation**

Create `src/vieneu/mlx/engine.py`:

```python
"""High-level MLX engine for VieNeu-TTS v3 Turbo.

Wraps the low-level MLX port (v3turbo_pipeline.V3TurboPipelineMLX) with the
same .infer(phonemes=..., speaker_emb=..., ref_codes=..., style=..., ...)
contract that _v3_turbo_engine.VieNeuTTSv3Turbo (PyTorch) and
_v3_turbo_engine.onnx_runtime_lite.OnnxV3LiteEngine (ONNX) already expose,
so it drops straight into V3TurboVieNeuTTS as backend="mlx" with no changes
to that class's chunking/gap-joining/watermark/preset-voice logic.

Preset voices only -- no runtime voice cloning (prepare_reference /
encode_reference are not implemented here). See
vieneu-tts-mlx-conversion-research-en.md section 1: the target use case
(reading from pre-existing voice presets) never needs to encode reference
audio at runtime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from transformers import AutoTokenizer

from vieneu._v3_turbo_engine.configuration_v3_turbo import VieNeuV3TurboConfig
from vieneu_utils.phonemize_text import phonemize_text_with_emotions

from .v3turbo_pipeline import V3TurboPipelineMLX


class MLXV3TurboEngine:
    """MLX backend for VieNeu-TTS v3 Turbo (preset voices only)."""

    def __init__(
        self,
        backbone_repo: str,
        model_subfolder: Optional[str],
        moss_tokenizer: str,
        backbone_weights: Union[str, Path],
        moss_weights: Union[str, Path],
        quantize_bits: Optional[int] = 4,
        quantize_group_size: int = 64,
    ):
        if not backbone_weights or not Path(backbone_weights).exists():
            raise FileNotFoundError(
                f"MLX backbone weights not found: {backbone_weights!r}. "
                "Convert them first with vieneu.mlx.convert_v3turbo_backbone."
            )
        if not moss_weights or not Path(moss_weights).exists():
            raise FileNotFoundError(
                f"MLX MOSS decoder weights not found: {moss_weights!r}. "
                "Convert them first with vieneu.mlx.convert_moss_decoder."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone_repo, subfolder=model_subfolder or "", trust_remote_code=True
        )
        self.config = VieNeuV3TurboConfig.from_pretrained(
            backbone_repo, subfolder=model_subfolder or ""
        )
        self.pipeline = V3TurboPipelineMLX(
            backbone_weights, moss_weights,
            quantize_bits=quantize_bits, quantize_group_size=quantize_group_size,
        )

    def _resolve_style_id(self, style: Union[int, str]) -> int:
        if isinstance(style, int):
            return style
        labels = getattr(self.config, "style_labels", None) or {}
        return labels.get(style, self.config.default_style_token_id)

    def infer(
        self,
        phonemes: Optional[str] = None,
        text: Optional[str] = None,
        ref_codes: Optional[np.ndarray] = None,
        speaker_emb: Optional[np.ndarray] = None,
        style: Union[int, str] = "tu_nhien",
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        **kwargs,
    ) -> np.ndarray:
        if phonemes is None:
            phonemes = phonemize_text_with_emotions(text or "")
        style_id = self._resolve_style_id(style)
        codes = self.pipeline.generate_codes(
            phonemes=phonemes,
            tokenizer=self.tokenizer,
            config=self.config,
            ref_codes=ref_codes if use_ref_codes else None,
            speaker_emb=speaker_emb,
            style_token_id=style_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_frames=max_new_frames,
        )
        if codes.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        return self.pipeline.synthesize(codes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Documents/git/VieNeu-TTS && .venv/bin/python -m pytest tests/test_mlx_engine.py -v`
Expected: PASS (6 tests: `test_missing_backbone_weights_raises`, `test_missing_moss_weights_raises`, `test_resolve_style_id_known_name`, `test_resolve_style_id_unknown_falls_back_to_default`, `test_resolve_style_id_int_passthrough`, `test_infer_calls_pipeline_generate_and_synthesize`, `test_infer_empty_codes_returns_empty_array`)

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/git/VieNeu-TTS
git add src/vieneu/mlx/engine.py tests/test_mlx_engine.py
git commit -m "feat(mlx): add MLXV3TurboEngine, a high-level infer()-compatible wrapper around V3TurboPipelineMLX"
```

---

## Task 2: Wire `backend="mlx"` into `V3TurboVieNeuTTS`

**Repo:** `~/Documents/git/VieNeu-TTS`

**Files:**
- Modify: `src/vieneu/v3turbo.py:30-87`
- Test: `tests/test_v3turbo_backend_mlx.py`

**Interfaces:**
- Consumes: `MLXV3TurboEngine` from Task 1 (imported lazily inside `__init__`, same pattern as the existing `OnnxV3LiteEngine`/`VieNeuTTSv3Turbo` lazy imports).
- Produces: `V3TurboVieNeuTTS(..., backend="mlx", mlx_backbone_weights: Optional[str] = None, mlx_moss_weights: Optional[str] = None, mlx_quantize_bits: Optional[int] = 4)` — sets `self.engine` to an `MLXV3TurboEngine` and `self.backend = "mlx"`. Consumed by Task 3 (robot-esp32's `vieneu_server.py`, via `Vieneu(mode="v3turbo", backend="mlx", ...)`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_v3turbo_backend_mlx.py`:

```python
"""Test that V3TurboVieNeuTTS(backend="mlx") builds an MLXV3TurboEngine."""
import sys
from unittest.mock import MagicMock, patch

# torch isn't needed for the mlx path, but v3turbo.py's device-resolution
# block does `import torch` unconditionally when device="auto" -- mock it
# the same way tests/test_factory.py does, so this test doesn't require a
# real torch install.
mock_torch = MagicMock()
mock_torch.Tensor = MagicMock
sys.modules.setdefault("torch", mock_torch)
sys.modules.setdefault("torch.backends", mock_torch.backends)
sys.modules.setdefault("torch.backends.mps", mock_torch.backends.mps)

from vieneu.v3turbo import V3TurboVieNeuTTS


@patch("vieneu.mlx.engine.MLXV3TurboEngine")
def test_v3turbo_backend_mlx_constructs_engine(mock_engine_cls):
    mock_engine_cls.return_value = MagicMock()
    tts = V3TurboVieNeuTTS(
        backend="mlx",
        mlx_backbone_weights="/tmp/backbone.safetensors",
        mlx_moss_weights="/tmp/moss.safetensors",
        mlx_quantize_bits=8,
    )
    assert tts.backend == "mlx"
    mock_engine_cls.assert_called_once_with(
        backbone_repo="pnnbao-ump/VieNeu-TTS-v3-Turbo",
        model_subfolder="update",
        moss_tokenizer="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        backbone_weights="/tmp/backbone.safetensors",
        moss_weights="/tmp/moss.safetensors",
        quantize_bits=8,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Documents/git/VieNeu-TTS && .venv/bin/python -m pytest tests/test_v3turbo_backend_mlx.py -v`
Expected: FAIL — `backend="mlx"` isn't handled, so `V3TurboVieNeuTTS.__init__` falls into the `else` (pytorch) branch and tries to construct a real `_v3_turbo_engine.VieNeuTTSv3Turbo`, which will error or, if it somehow doesn't, `mock_engine_cls.assert_called_once_with(...)` fails because it was never called.

- [ ] **Step 3: Implement the `backend="mlx"` branch**

In `src/vieneu/v3turbo.py`, modify the `__init__` signature (around line 30-44) — replace:

```python
        backend: str = "auto",   # "auto" → ONNX on CPU, PyTorch on GPU; "onnx"|"pytorch" to force
        onnx_repo: Optional[str] = None,
        onnx_dir: Optional[str] = None,
        precision: str = "int8",   # ONNX/CPU backbone: "int8" (mặc định, nhanh ~3x/frame, nhỏ 4x) | "fp32" (chất-lượng-tối-đa)
        onnx_subfolder: Optional[str] = None,   # override thủ công subfolder; None → suy từ `precision`
        threads: int = 0,   # ONNX/CPU intra-op threads; 0 = mặc định engine (~nhân vật lý, cap 8). Đặt số cụ thể để tinh chỉnh.
        **kwargs: Any,
    ):
```

with:

```python
        backend: str = "auto",   # "auto" → ONNX on CPU, PyTorch on GPU; "onnx"|"pytorch"|"mlx" to force
        onnx_repo: Optional[str] = None,
        onnx_dir: Optional[str] = None,
        precision: str = "int8",   # ONNX/CPU backbone: "int8" (mặc định, nhanh ~3x/frame, nhỏ 4x) | "fp32" (chất-lượng-tối-đa)
        onnx_subfolder: Optional[str] = None,   # override thủ công subfolder; None → suy từ `precision`
        threads: int = 0,   # ONNX/CPU intra-op threads; 0 = mặc định engine (~nhân vật lý, cap 8). Đặt số cụ thể để tinh chỉnh.
        mlx_backbone_weights: Optional[str] = None,   # backend="mlx" only: path to the converted backbone .safetensors
        mlx_moss_weights: Optional[str] = None,       # backend="mlx" only: path to the converted MOSS decoder .safetensors
        mlx_quantize_bits: Optional[int] = 4,          # backend="mlx" only: None=fp32, 8=q8, 4=q4 (default)
        **kwargs: Any,
    ):
```

Then replace the engine-construction block (around line 61-87) — replace:

```python
        use_onnx = backend == "onnx" or (backend == "auto" and dev_type == "cpu")

        if use_onnx:
            # Torch-free CPU engine. Reads its ONNX graphs from `onnx_subfolder` in the
            # model repo (uploaded separately).
            from ._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (ONNX/CPU) from: {backbone_repo}/{onnx_subfolder} ...")
            self.engine = OnnxV3LiteEngine(
                checkpoint_path=backbone_repo,
                onnx_repo=onnx_repo,
                onnx_dir=onnx_dir,
                onnx_subfolder=onnx_subfolder,
                threads=threads,
            )
            self.backend = "onnx"
        else:
            from ._v3_turbo_engine import VieNeuTTSv3Turbo
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (PyTorch) from: {backbone_repo}/{model_subfolder} ...")
            self.engine = VieNeuTTSv3Turbo(
                checkpoint_path=backbone_repo,
                model_subfolder=model_subfolder,
                moss_tokenizer_path=moss_tokenizer,
                device=device,
                dtype=dtype,
            )
            self.backend = "pytorch"
        logger.info(f"✅ VieNeu-TTS v3 Turbo ready (backend={self.backend})")
```

with:

```python
        use_onnx = backend == "onnx" or (backend == "auto" and dev_type == "cpu")

        if backend == "mlx":
            from .mlx.engine import MLXV3TurboEngine
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (MLX) from: {backbone_repo}/{model_subfolder} ...")
            self.engine = MLXV3TurboEngine(
                backbone_repo=backbone_repo,
                model_subfolder=model_subfolder,
                moss_tokenizer=moss_tokenizer,
                backbone_weights=mlx_backbone_weights,
                moss_weights=mlx_moss_weights,
                quantize_bits=mlx_quantize_bits,
            )
            self.backend = "mlx"
        elif use_onnx:
            # Torch-free CPU engine. Reads its ONNX graphs from `onnx_subfolder` in the
            # model repo (uploaded separately).
            from ._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (ONNX/CPU) from: {backbone_repo}/{onnx_subfolder} ...")
            self.engine = OnnxV3LiteEngine(
                checkpoint_path=backbone_repo,
                onnx_repo=onnx_repo,
                onnx_dir=onnx_dir,
                onnx_subfolder=onnx_subfolder,
                threads=threads,
            )
            self.backend = "onnx"
        else:
            from ._v3_turbo_engine import VieNeuTTSv3Turbo
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (PyTorch) from: {backbone_repo}/{model_subfolder} ...")
            self.engine = VieNeuTTSv3Turbo(
                checkpoint_path=backbone_repo,
                model_subfolder=model_subfolder,
                moss_tokenizer_path=moss_tokenizer,
                device=device,
                dtype=dtype,
            )
            self.backend = "pytorch"
        logger.info(f"✅ VieNeu-TTS v3 Turbo ready (backend={self.backend})")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Documents/git/VieNeu-TTS && .venv/bin/python -m pytest tests/test_v3turbo_backend_mlx.py -v`
Expected: PASS

- [ ] **Step 5: Run the full existing test suite to check for regressions**

Run: `cd ~/Documents/git/VieNeu-TTS && .venv/bin/python -m pytest tests/ -v`
Expected: All tests pass (same pass/fail set as before this change, plus the new tests from Tasks 1-2). No pre-existing test should newly fail — `use_onnx`'s definition and the `elif`/`else` branches are behaviorally identical to the prior `if`/`else` for every `backend` value except the new `"mlx"`.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/git/VieNeu-TTS
git add src/vieneu/v3turbo.py tests/test_v3turbo_backend_mlx.py
git commit -m "feat(v3turbo): add backend=\"mlx\" option using MLXV3TurboEngine"
```

---

## Task 3: Bump package version and install editable into robot-esp32's services venv

**Repo:** `~/Documents/git/VieNeu-TTS` (edit) and `~/Documents/git/robot-esp32` (edit + install)

**Files:**
- Modify: `~/Documents/git/VieNeu-TTS/pyproject.toml:36` and `:70`
- Modify: `~/Documents/git/robot-esp32/services/requirements.txt`

**Interfaces:**
- Produces: `services/.venv` with `vieneu` importable as an editable install exposing `vieneu.mlx.engine.MLXV3TurboEngine` and `Vieneu(mode="v3turbo", backend="mlx", ...)`. Consumed by Task 4.

- [ ] **Step 1: Bump the VieNeu-TTS package version and sea-g2p floor**

In `~/Documents/git/VieNeu-TTS/pyproject.toml`, change:

```toml
version = "3.2.0"
```

to:

```toml
version = "3.2.3"
```

and change:

```toml
    "sea-g2p>=0.7.19",      # phonemizer (Rust, torch-free) — cần hàm punc_norm() (>=0.7.11)
```

to:

```toml
    "sea-g2p>=0.7.20",      # phonemizer (Rust, torch-free) — cần hàm punc_norm() (>=0.7.11)
```

(These are the only two lines that changed between the local checkout — commit `5f758da`, 13 commits behind the real `v3.2.3` tag — and `v3.2.3` that matter for this server; the rest of that gap is web-demo streaming fixes, GPU auto-batch packaging, and docs. Per the global constraints, git history is not touched — this is a plain file edit on top of the existing uncommitted changes.)

- [ ] **Step 2: Point robot-esp32's requirements.txt at the local editable checkout**

In `~/Documents/git/robot-esp32/services/requirements.txt`, replace:

```
vieneu==3.0.5
```

with:

```
# Editable install from the local checkout (not PyPI): the MLX backend
# (src/vieneu/mlx/) is uncommitted local work, not published. Content is
# vieneu 3.2.3 (see that repo's pyproject.toml) + the MLX additions.
-e /Users/lucnguyen/Documents/git/VieNeu-TTS[gpu,mlx]
```

- [ ] **Step 3: Install into the services venv**

Run:
```bash
cd ~/Documents/git/robot-esp32/services
.venv/bin/pip install -e "/Users/lucnguyen/Documents/git/VieNeu-TTS[gpu,mlx]"
```
Expected: Installs/upgrades `vieneu` (editable), `mlx-lm` (new), `sea-g2p` (0.7.6 → ≥0.7.20), and whatever `torch`/`transformers`/`neucodec`/`llama-cpp-python` versions the `[gpu]` extra resolves to. `mlx` (already 0.31.2) should be left alone since it already satisfies `>=0.20`. If pip reports a version conflict against another already-pinned package in `requirements.txt`, resolve it by installing the version pip picks (don't force-pin backward) — this file is a `pip freeze` snapshot, not a hand-authored constraint set.

- [ ] **Step 4: Verify the editable install is importable and exposes the new backend**

Run:
```bash
cd ~/Documents/git/robot-esp32/services
.venv/bin/python -c "
import vieneu
from vieneu.mlx.engine import MLXV3TurboEngine
from vieneu.v3turbo import V3TurboVieNeuTTS
import mlx_lm
print('vieneu OK, mlx_lm OK')
"
```
Expected: prints `vieneu OK, mlx_lm OK` with no `ImportError`/`ModuleNotFoundError`.

- [ ] **Step 5: Confirm the shared venv's other service (Whisper) still imports cleanly**

`services/.venv` is shared with `whisper_server.py` (per `services/README.md`), which also depends on `transformers`/`mlx`. Since the `[gpu]` extra may have changed `transformers`'s pinned version, check it didn't break:
```bash
cd ~/Documents/git/robot-esp32/services
.venv/bin/python -c "import mlx, transformers; print(mlx.core.__name__, transformers.__version__)"
```
Expected: no import error. (Full Whisper startup is checked in Task 5's end-to-end verification.)

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/git/VieNeu-TTS
git add pyproject.toml
git commit -m "chore: bump version to 3.2.3, require sea-g2p>=0.7.20 (matches upstream v3.2.3)"

cd ~/Documents/git/robot-esp32
git add services/requirements.txt
git commit -m "$(cat <<'EOF'
chore(services): install vieneu editable from local checkout, not PyPI 3.0.5

The MLX backend only exists in the local VieNeu-TTS checkout (uncommitted
research work), so vieneu_server.py's upcoming MLX default needs an
editable install rather than the stale PyPI 3.0.5 pin.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UtXgNVeQr1KwnKEZAeGUaT
EOF
)"
```

---

## Task 4: `VIENEU_BACKEND` switch in `vieneu_server.py`

**Repo:** `~/Documents/git/robot-esp32`

**Files:**
- Modify: `services/vieneu_server.py:30-181` (env vars + engine construction) and `:292-296` (`/health`)

**Interfaces:**
- Consumes: `Vieneu(mode="v3turbo", backend=..., mlx_backbone_weights=..., mlx_moss_weights=..., mlx_quantize_bits=..., device=...)` from Task 3's installed package.

- [ ] **Step 1: Add the new env vars**

In `services/vieneu_server.py`, after line 35 (`EMOTION = os.environ.get(...)`), add:

```python
# v3turbo only: which engine backs Vieneu(mode="v3turbo"). "mlx" (default) = the MLX port
# (Apple Silicon, see vieneu-tts-mlx-conversion-research-en.md). "pytorch" = force MPS (the
# previous default here). "onnx" = the package's CPU/int8 engine. Rollback: set to "pytorch".
BACKEND = os.environ.get("VIENEU_BACKEND", "mlx")
MLX_BACKBONE_WEIGHTS = os.environ.get("VIENEU_MLX_BACKBONE_WEIGHTS", "/Volumes/Data/vieneu-mlx/v3turbo_backbone.safetensors")
MLX_MOSS_WEIGHTS = os.environ.get("VIENEU_MLX_MOSS_WEIGHTS", "/Volumes/Data/vieneu-mlx/moss_decoder.safetensors")
_mlx_quant = os.environ.get("VIENEU_MLX_QUANTIZE", "4").strip()
MLX_QUANTIZE_BITS = int(_mlx_quant) if _mlx_quant else None  # "" or "0" -> fp32
```

- [ ] **Step 2: Replace the v3turbo engine construction**

Replace (around line 169-180):

```python
log(f"nạp VieNeu-TTS (mode={MODE}, giọng {VOICE})...")
# turbo does NOT accept an emotion parameter (only standard does).
if MODE == "standard":
    tts = Vieneu(mode=MODE, emotion=EMOTION)
elif MODE == "v3turbo":
    # v3turbo's own device="auto" only checks torch.cuda (not available on Mac) -> it ALWAYS
    # falls back to CPU/ONNX (onnx_runtime_lite.py hardcodes CPUExecutionProvider), ignoring the
    # GPU MPS that's actually available -> much slower (~6-7s/sentence instead of ~3.3s). Force
    # device=mps to use PyTorch+Metal instead.
    tts = Vieneu(mode=MODE, device="mps")
else:
    tts = Vieneu(mode=MODE)
```

with:

```python
log(f"nạp VieNeu-TTS (mode={MODE}, backend={BACKEND}, giọng {VOICE})...")
# turbo does NOT accept an emotion parameter (only standard does).
if MODE == "standard":
    tts = Vieneu(mode=MODE, emotion=EMOTION)
elif MODE == "v3turbo":
    if BACKEND == "mlx":
        tts = Vieneu(
            mode=MODE, backend="mlx",
            mlx_backbone_weights=MLX_BACKBONE_WEIGHTS,
            mlx_moss_weights=MLX_MOSS_WEIGHTS,
            mlx_quantize_bits=MLX_QUANTIZE_BITS,
        )
    elif BACKEND == "pytorch":
        # v3turbo's own device="auto" only checks torch.cuda (not available on Mac) -> it ALWAYS
        # falls back to CPU/ONNX (onnx_runtime_lite.py hardcodes CPUExecutionProvider), ignoring the
        # GPU MPS that's actually available -> much slower (~6-7s/sentence instead of ~3.3s). Force
        # device=mps to use PyTorch+Metal instead.
        tts = Vieneu(mode=MODE, device="mps")
    else:
        tts = Vieneu(mode=MODE, backend=BACKEND)
else:
    tts = Vieneu(mode=MODE)
```

- [ ] **Step 3: Report the backend in `/health`**

Replace (around line 292-296):

```python
@app.get("/health")
def health():
    keys = list(getattr(tts, "_preset_voices", {}).keys())
    return {"status": "ok", "voice": VOICE, "mode": MODE, "emotion": EMOTION,
            "voices": keys, "cached": list(_voice_cache.keys())}
```

with:

```python
@app.get("/health")
def health():
    keys = list(getattr(tts, "_preset_voices", {}).keys())
    return {"status": "ok", "voice": VOICE, "mode": MODE, "backend": BACKEND, "emotion": EMOTION,
            "voices": keys, "cached": list(_voice_cache.keys())}
```

- [ ] **Step 4: Byte-compile check (no test infra exists for this file — it has module-level startup side effects)**

Run: `cd ~/Documents/git/robot-esp32/services && .venv/bin/python -m py_compile vieneu_server.py`
Expected: no output, exit code 0 (syntax/name-resolution sanity check only — real behavior is verified in Task 6).

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/git/robot-esp32
git add services/vieneu_server.py
git commit -m "$(cat <<'EOF'
feat(vieneu): add VIENEU_BACKEND switch (mlx default, pytorch/onnx rollback)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UtXgNVeQr1KwnKEZAeGUaT
EOF
)"
```

---

## Task 5: Docs — `run_vieneu.sh` and `services/README.md`

**Repo:** `~/Documents/git/robot-esp32`

**Files:**
- Modify: `services/run_vieneu.sh`
- Modify: `services/README.md:15`

**Interfaces:** None (docs only).

- [ ] **Step 1: Document the rollback knob in the launcher**

In `services/run_vieneu.sh`, after the `VIENEU_MODE` line, add:

```bash
export VIENEU_BACKEND=mlx                          # v3turbo only. mlx (default, Apple Silicon MLX port, ~q4). Rollback: pytorch (force MPS) or onnx (CPU int8).
```

- [ ] **Step 2: Update the services README row**

In `services/README.md`, replace the VieNeu row (line 15):

```
| **VieNeu (TTS)** | `vieneu_server.py` | `run_vieneu.sh` | 8002 | `.venv` | Vietnamese text-to-speech (multiple voices). Required for the voice loop. |
```

with:

```
| **VieNeu (TTS)** | `vieneu_server.py` | `run_vieneu.sh` | 8002 | `.venv` | Vietnamese text-to-speech (multiple voices). Required for the voice loop. v3turbo mode defaults to the MLX backend (`VIENEU_BACKEND=mlx`); set `pytorch`/`onnx` to roll back. |
```

- [ ] **Step 3: Commit**

```bash
cd ~/Documents/git/robot-esp32
git add services/run_vieneu.sh services/README.md
git commit -m "$(cat <<'EOF'
docs(services): document VIENEU_BACKEND rollback knob

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UtXgNVeQr1KwnKEZAeGUaT
EOF
)"
```

---

## Task 6: End-to-end verification

**Repo:** `~/Documents/git/robot-esp32`

**Files:** None modified — manual verification only, since this file has no test harness.

- [ ] **Step 1: Start the server on the MLX backend (default) on a scratch port**

```bash
cd ~/Documents/git/robot-esp32/services
source .venv/bin/activate
VIENEU_MODE=v3turbo VIENEU_BACKEND=mlx VIENEU_PORT=8099 VIENEU_VOICE="Ngọc Lan" python vieneu_server.py &
sleep 2  # let it boot the model before curling
```

- [ ] **Step 2: Confirm `/health` reports the mlx backend**

```bash
curl -s http://localhost:8099/health
```
Expected: JSON with `"mode": "v3turbo"`, `"backend": "mlx"`, and `"voices"` containing `"Ngọc Lan"`.

- [ ] **Step 3: Confirm `/tts` returns real (non-silent) audio**

```bash
curl -s -X POST http://localhost:8099/tts -H "Content-Type: application/json" \
  -d '{"input": "Xin chào, đây là bản thử nghiệm giọng nói MLX."}' -o /tmp/vieneu_mlx_test.wav
ls -la /tmp/vieneu_mlx_test.wav
```
Expected: a WAV file well over the ~7KB `SILENT_WAV` floor (a few seconds of 48kHz 16-bit mono should be several hundred KB). Play it (e.g. `afplay /tmp/vieneu_mlx_test.wav`) and confirm it's audible, correct-sounding Vietnamese speech, not silence or noise.

- [ ] **Step 4: Stop the server, then repeat for the pytorch rollback**

```bash
kill %1 2>/dev/null; wait 2>/dev/null
VIENEU_MODE=v3turbo VIENEU_BACKEND=pytorch VIENEU_PORT=8099 VIENEU_VOICE="Ngọc Lan" python vieneu_server.py &
sleep 2
curl -s http://localhost:8099/health   # expect "backend": "pytorch"
curl -s -X POST http://localhost:8099/tts -H "Content-Type: application/json" \
  -d '{"input": "Xin chào, đây là bản thử nghiệm giọng nói PyTorch."}' -o /tmp/vieneu_pytorch_test.wav
ls -la /tmp/vieneu_pytorch_test.wav   # expect similarly-sized non-silent WAV
kill %1 2>/dev/null; wait 2>/dev/null
```

- [ ] **Step 5: Repeat for the onnx backend**

```bash
VIENEU_MODE=v3turbo VIENEU_BACKEND=onnx VIENEU_PORT=8099 VIENEU_VOICE="Ngọc Lan" python vieneu_server.py &
sleep 2
curl -s http://localhost:8099/health   # expect "backend": "onnx"
curl -s -X POST http://localhost:8099/tts -H "Content-Type: application/json" \
  -d '{"input": "Xin chào, đây là bản thử nghiệm giọng nói ONNX."}' -o /tmp/vieneu_onnx_test.wav
ls -la /tmp/vieneu_onnx_test.wav
kill %1 2>/dev/null; wait 2>/dev/null
```

- [ ] **Step 6: Sanity-check `MODE=standard` (v2 GGUF) still boots on the upgraded package**

```bash
VIENEU_MODE=standard VIENEU_PORT=8099 VIENEU_VOICE=Doan FILLER_REGEN_ON_VOICE=0 python vieneu_server.py &
sleep 2
curl -s http://localhost:8099/health   # expect "mode": "standard"
curl -s -X POST http://localhost:8099/tts -H "Content-Type: application/json" \
  -d '{"input": "Xin chào, đây là bản thử nghiệm giọng nói tiêu chuẩn."}' -o /tmp/vieneu_standard_test.wav
ls -la /tmp/vieneu_standard_test.wav
kill %1 2>/dev/null; wait 2>/dev/null
```

- [ ] **Step 7: If all four checks pass, restart production with the new default**

```bash
cd ~/Documents/git/robot-esp32/services
./run_vieneu.sh   # now defaults to VIENEU_BACKEND=mlx per Task 5
```
Confirm via the existing project workflow for checking the live robot voice (see the `restart-xiaozhi-server` memory note) that the assistant still speaks correctly end-to-end.

- [ ] **Step 8: Clean up scratch files**

```bash
rm -f /tmp/vieneu_mlx_test.wav /tmp/vieneu_pytorch_test.wav /tmp/vieneu_onnx_test.wav /tmp/vieneu_standard_test.wav
```

---

## Self-Review Notes

- **Spec coverage:** Task 1-2 cover the spec's §1 (MLXV3TurboEngine + `v3turbo.py` wiring); Task 3 covers §1's `pyproject.toml` bump and §2's `requirements.txt` editable install; Task 4 covers §2's `vieneu_server.py` env vars/dispatch/`/health`; Task 5 covers §2's `run_vieneu.sh`/README docs; Task 6 covers §3's verification steps (mlx/pytorch/onnx/standard, all four spec bullets).
- **Non-goals honored:** no voice-cloning code added to `MLXV3TurboEngine`; no `infer_stream` implementation (existing fallback in `V3TurboVieNeuTTS.infer_stream` handles it); no git sync/push/tag of VieNeu-TTS; `MODE=standard` code path untouched (only exercised for regression-checking in Task 6).
- **Type/signature consistency:** `MLXV3TurboEngine.__init__` params (Task 1) match exactly what `v3turbo.py`'s new branch passes (Task 2) match exactly what `vieneu_server.py` passes (Task 4) — `backbone_weights`/`moss_weights`/`quantize_bits` used consistently throughout, no renamed fields.
