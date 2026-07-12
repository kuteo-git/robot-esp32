"""Download RIR and background-noise datasets for training-time augmentation.

Mirrors vendor/microWakeWord/notebooks/basic_training_notebook.ipynb's cell 4
verbatim (dataset repo IDs, filenames, and the flac/mp3 -> 16kHz WAV conversion),
producing directories consumable by extract_features.py's --rir-dir/--background-dir:

    data/mit_rirs/       <- --rir-dir     (room impulse responses)
    data/audioset_16k/   <- --background-dir (general background noise)
    data/fma_16k/        <- --background-dir (music, as background noise)

NOTE (from the notebook, verbatim): "the data downloaded here has a mixture of
different licenses and usage restrictions. As such, any custom models trained
with this data should be considered as appropriate for non-commercial personal
use only."

The notebook itself only downloads one segment of AudioSet ("bal_train09.tar")
as a quickstart-sized example, noting "for full-scale training, it's
recommended to download the entire dataset" -- this script matches that same
scope (a fixed-size subset, not the full ~22k-clip balanced split) rather than
inventing a larger one. NOTE: the notebook's literal tar-shard URL
(agkphysics/AudioSet/resolve/main/data/bal_train09.tar) is defunct -- that
repo has since been restructured into proper HuggingFace Parquet format
(data/bal_train/*.parquet, confirmed via the HF API, not guessed), so this
downloads the same "balanced train" split via `datasets.load_dataset` and
takes a fixed-size prefix instead of a fixed tar shard.

Must be run with services/wakeword_training/.venv-train/bin/python (needs
`datasets`/`scipy`, already installed via microWakeWord's own dependencies):
    services/wakeword_training/.venv-train/bin/python \
        services/wakeword_training/download_augmentation_data.py --out-dir data
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile

MIT_RIR_REPO_ID = "davidscripka/MIT_environmental_impulse_responses"
AUDIOSET_REPO_ID = "agkphysics/AudioSet"
AUDIOSET_CLIP_LIMIT = 1000  # quickstart-sized subset, matching one old tar shard's rough scale
FMA_ZIP_URL = "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"


def _write_16k_wav(out_dir: Path, name: str, audio: np.ndarray) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(out_dir / name), 16000, (audio * 32767).astype(np.int16))


def download_mit_rirs(out_dir: Path) -> int:
    out_dir = Path(out_dir)
    existing = list(out_dir.glob("*.wav")) if out_dir.exists() else []
    if existing:
        return len(existing)

    import datasets

    rir_dataset = datasets.load_dataset(MIT_RIR_REPO_ID, split="train", streaming=True)
    count = 0
    for row in rir_dataset:
        name = row["audio"]["path"].split("/")[-1]
        _write_16k_wav(out_dir, name, np.asarray(row["audio"]["array"], dtype=np.float32))
        count += 1
    return count


def download_audioset(out_dir: Path, limit: int = AUDIOSET_CLIP_LIMIT) -> int:
    out_dir = Path(out_dir)
    existing = list(out_dir.glob("*.wav")) if out_dir.exists() else []
    if existing:
        return len(existing)

    import datasets

    audioset_dataset = datasets.load_dataset(
        AUDIOSET_REPO_ID, "balanced", split="train", streaming=True
    )
    audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    count = 0
    for row in audioset_dataset:
        if count >= limit:
            break
        name = f"{row['video_id']}.wav"
        _write_16k_wav(out_dir, name, np.asarray(row["audio"]["array"], dtype=np.float32))
        count += 1
    return count


def download_fma(work_dir: Path, out_dir: Path) -> int:
    out_dir = Path(out_dir)
    existing = list(out_dir.glob("*.wav")) if out_dir.exists() else []
    if existing:
        return len(existing)

    import datasets

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    zip_path = work_dir / "fma_xs.zip"
    if not zip_path.exists():
        subprocess.run(["curl", "-L", "-o", str(zip_path), FMA_ZIP_URL], check=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)

    mp3_paths = [str(p) for p in work_dir.glob("**/*.mp3")]
    fma_dataset = datasets.Dataset.from_dict({"audio": mp3_paths})
    fma_dataset = fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    count = 0
    for row in fma_dataset:
        name = Path(row["audio"]["path"]).stem + ".wav"
        _write_16k_wav(out_dir, name, np.asarray(row["audio"]["array"], dtype=np.float32))
        count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    rir_count = download_mit_rirs(out_dir / "mit_rirs")
    print(f"MIT RIR: {rir_count} clips -> {out_dir / 'mit_rirs'}")

    audioset_count = download_audioset(out_dir / "audioset_16k")
    print(f"AudioSet (balanced train, first {AUDIOSET_CLIP_LIMIT}): {audioset_count} clips -> {out_dir / 'audioset_16k'}")

    fma_count = download_fma(out_dir / "_fma_raw", out_dir / "fma_16k")
    print(f"FMA xs: {fma_count} clips -> {out_dir / 'fma_16k'}")


if __name__ == "__main__":
    main()
