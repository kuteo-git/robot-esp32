# VieNeu-TTS MLX backend for vieneu_server.py

## Context

`services/vieneu_server.py` runs the robot's Vietnamese TTS. Production (`run_vieneu.sh`) uses
`VIENEU_MODE=v3turbo` (48kHz, `pnnbao-ump/VieNeu-TTS-v3-Turbo`), which today loads either the
package's ONNX-int8 CPU engine or a PyTorch/MPS engine (forced via `device="mps"` in
`vieneu_server.py` because the package's own `device="auto"` only checks `torch.cuda`, never MPS).

The user has already done the MLX conversion research and implementation in a separate repo,
`~/Documents/git/VieNeu-TTS` (uncommitted): `src/vieneu/mlx/{v3turbo_backbone,moss_decoder,
v3turbo_pipeline}.py` port the v3 Turbo backbone (Qwen3 + AcousticDecoder) and the
MOSS-Audio-Tokenizer-Nano decoder to MLX, bit-exact vs. PyTorch, with backbone quantization
benchmarked against production (see `vieneu-tts-mlx-conversion-research-en.md` §8). Converted
weights already exist on disk at `/Volumes/Data/vieneu-mlx/{v3turbo_backbone,moss_decoder}.safetensors`.

That MLX code is low-level research code — `V3TurboPipelineMLX` only has `generate_codes(...)` /
`synthesize(codes)`, not the `infer(text, voice=...)` / `save()` / preset-voice API
`vieneu_server.py` needs. This design adds a thin high-level wrapper and wires it into the server
as an opt-in backend, alongside the existing ones, so the server can roll back with one env var.

**Decisions already made with the user (see conversation):**
- Do **not** git-sync the VieNeu-TTS checkout to the `v3.2.3` tag (13 commits behind, all
  irrelevant — web-demo streaming fixes, GPU auto-batch, docs, version-bump chores). Work from the
  current checkout + its uncommitted MLX files as-is.
- The MLX adapter lives inside VieNeu-TTS's `src/vieneu/mlx/` (reusable, consistent with the
  package's existing `backend="onnx"|"pytorch"` selector), not duplicated into robot-esp32.
- Default MLX precision: **q4** quantization (backbone Linear layers only, matching the research
  doc's own benchmark setup).

## Non-goals

- No voice cloning (`add_voice`/`encode_reference`) on the MLX backend — preset voices only,
  matching the research doc's stated scope (§1: "does not need to encode reference audio at
  runtime"). If cloning is ever needed, fall back to `pytorch`/`onnx`.
- No streaming (`infer_stream`) implementation for MLX — `V3TurboVieNeuTTS.infer_stream` already
  falls back to non-streaming `.infer()` per chunk when the engine lacks `infer_stream`, and
  `vieneu_server.py` doesn't use streaming today.
- No PyPI publish, no `git push`/tag of VieNeu-TTS. This stays a local editable install.
- `MODE=standard` (true v2 GGUF) is untouched — the "GGUF/PyTorch" upgrade in this task refers to
  bumping the whole `vieneu` package (which also covers v3turbo's onnx/pytorch engines, the ones
  actually in production use), not adding an MLX path to v2.

## Design

### 1. VieNeu-TTS repo (`~/Documents/git/VieNeu-TTS`)

**New file `src/vieneu/mlx/engine.py`** — `MLXV3TurboEngine`:
- `__init__(backbone_repo, model_subfolder, moss_tokenizer, backbone_weights, moss_weights, quantize_bits=4, quantize_group_size=64)`:
  loads `AutoTokenizer`/`VieNeuV3TurboConfig` from `backbone_repo` (HF, same repo the PyTorch
  engine uses — only the tokenizer/config, not the PyTorch weights), then
  `V3TurboPipelineMLX(backbone_weights, moss_weights, quantize_bits=...)`. Raises
  `FileNotFoundError` with a clear message (pointing at `convert_v3turbo_backbone.py`/
  `convert_moss_decoder.py`) if the weight files are missing.
- `_resolve_style_id(style)`: same logic as `VieNeuTTSv3Turbo._resolve_style_id` (int passthrough,
  else `config.style_labels.get(style, config.default_style_token_id)`).
- `infer(phonemes=None, text=None, ref_codes=None, speaker_emb=None, style="tu_nhien",
  use_ref_codes=True, temperature=0.8, top_k=25, top_p=0.95, max_new_frames=300,
  repetition_penalty=1.2, **kwargs) -> np.ndarray`: phonemizes `text` if `phonemes` is `None`
  (reusing `vieneu_utils.phonemize_text.phonemize_text_with_emotions`, same as the PyTorch
  engine), resolves style, calls `pipeline.generate_codes(...)` then `pipeline.synthesize(codes)`.
  Returns `np.zeros(0, dtype=np.float32)` on empty codes (matches `_v3_turbo_engine`'s
  `_decode_codes` empty-input behavior).

This method signature exactly matches what `V3TurboVieNeuTTS.infer`/`.infer_stream` already call
on `self.engine`, so no changes are needed to the chunking/gap-joining/watermark/preset-voice logic
in `v3turbo.py` — only the engine construction.

**Edit `src/vieneu/v3turbo.py`** (`V3TurboVieNeuTTS.__init__`):
- New constructor kwargs: `mlx_backbone_weights: Optional[str] = None`,
  `mlx_moss_weights: Optional[str] = None`, `mlx_quantize_bits: Optional[int] = 4`.
- Branch order becomes: `if backend == "mlx": ... elif use_onnx: ... else: # pytorch`. `"auto"`
  keeps resolving to onnx-on-CPU/pytorch-on-GPU exactly as today — `mlx` is only selected when
  explicitly requested, so this is fully backward compatible for other users of the package.

**Edit `pyproject.toml`**: `version = "3.2.0"` → `"3.2.3"`; `"sea-g2p>=0.7.19"` →
`"sea-g2p>=0.7.20"` (the one dependency-relevant change in the 13-commit gap; everything else
there is web-demo/GPU-batching/docs and doesn't affect this server). The `mlx` extra
(`mlx>=0.20`, `mlx-lm>=0.20`) already exists in this file — no change needed there.

### 2. robot-esp32/services

**`requirements.txt`**: replace the `vieneu==3.0.5` line with
`-e /Users/lucnguyen/Documents/git/VieNeu-TTS[gpu,mlx]` plus a comment explaining why (local
editable install — the MLX code isn't published to PyPI). Keeps `[gpu]` (torch, neucodec,
llama-cpp-python, transformers, librosa, accelerate — needed for `MODE=standard` and
`backend=pytorch`) and adds `[mlx]` (mlx, mlx-lm).

**`vieneu_server.py`**:
- New env vars: `VIENEU_BACKEND` (default `"mlx"`, only consulted when `MODE=="v3turbo"`; other
  values `"pytorch"`, `"onnx"`), `VIENEU_MLX_BACKBONE_WEIGHTS` (default
  `/Volumes/Data/vieneu-mlx/v3turbo_backbone.safetensors`), `VIENEU_MLX_MOSS_WEIGHTS` (default
  `/Volumes/Data/vieneu-mlx/moss_decoder.safetensors`), `VIENEU_MLX_QUANTIZE` (default `"4"`;
  `"8"`/`"4"`/empty-string for fp32).
- The `MODE == "v3turbo"` branch dispatches on `BACKEND`: `"mlx"` builds
  `Vieneu(mode="v3turbo", backend="mlx", mlx_backbone_weights=..., mlx_moss_weights=...,
  mlx_quantize_bits=...)`; `"pytorch"` keeps today's `Vieneu(mode="v3turbo", device="mps")`;
  `"onnx"` (or anything else) passes `backend=BACKEND` through unchanged (package default CPU/int8
  behavior).
- `/health` response gains a `"backend"` field.

**`run_vieneu.sh`**: add a commented `VIENEU_BACKEND` line documenting the rollback (`mlx` default,
set to `pytorch` to roll back to the current production behavior).

**`services/README.md`**: update the VieNeu row / add a short note on the new env var, per this
repo's own rule that setup-affecting docs get updated in the same pass as the change.

### 3. Verification

1. `services/.venv/bin/pip install -r services/requirements.txt` (or targeted
   `pip install -e "...[gpu,mlx]"`) — confirm it resolves without conflicts (mlx-lm newly
   installed, sea-g2p 0.7.6→0.7.20+, torch/neucodec/llama-cpp-python versions may shift under the
   3.0.5→3.2.3 package jump).
2. Start the server with `VIENEU_BACKEND=mlx VIENEU_MODE=v3turbo`, `POST /tts` with a short
   Vietnamese sentence, confirm non-silent WAV back and reasonable latency in the log line.
3. Re-test with `VIENEU_BACKEND=pytorch` and `VIENEU_BACKEND=onnx` to confirm the package upgrade
   didn't break the existing rollback paths (preset voice resolution, `/voice` switching, filler
   regen).
4. Sanity-check `MODE=standard` (v2 GGUF) still boots, since it's on the same upgraded package.
