"""Download microWakeWord's standard background-noise/negative-speech datasets.

Mechanism and dataset identifiers below are taken directly from
vendor/microWakeWord/notebooks/basic_training_notebook.ipynb, cell 8
("Downloads pre-generated spectrogram features (made for microWakeWord in
particular) for various negative datasets. This can be slow!"):

    output_dir = './negative_datasets'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        link_root = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
        filenames = ['dinner_party.zip', 'dinner_party_eval.zip', 'no_speech.zip', 'speech.zip']
        for fname in filenames:
            link = link_root + fname
            zip_path = f"negative_datasets/{fname}"
            !wget -O {zip_path} {link}
            !unzip -q {zip_path} -d {output_dir}

The notebook fetches these 4 zip files with a raw `wget` (not
`huggingface_hub`) from the HF *dataset* repo `kahrendt/microwakeword`
(https://huggingface.co/datasets/kahrendt/microwakeword). We reproduce the
same downloads here using `huggingface_hub.hf_hub_download` instead of
shelling out to `wget`/`unzip`, for a portable, dependency-free-of-shell-tools
implementation that fetches the exact same files from the exact same repo.
This was corroborated live against the HF datasets API
(`https://huggingface.co/api/datasets/kahrendt/microwakeword`), whose
`siblings` list includes exactly these filenames (plus a few `*_background.zip`
files the notebook does not use) and whose repo description states: "This
dataset contains spectrogram features in an mmap ninja format intended to
use for microWakeWord training."

That description, and vendor/microWakeWord/microwakeword/data.py's own
`MmapFeatureGenerator` (which reads these datasets back), confirm the format
is **Ragged Mmap folders** (written by `mmap_ninja.ragged.RaggedMmap`), NOT
raw audio files (.wav/.flac/.ogg). `MmapFeatureGenerator.__init__` locates
these features with:

    search_path = [
        str(i)
        for i in Path(os.path.abspath(search_path_directory)).glob("**/*_mmap/")
    ]

i.e. it globs for directories whose name ends in "_mmap" underneath each of
the split directories ("training", "validation", "testing",
"validation_ambient", "testing_ambient"). `validate_dataset_dir` below
mirrors that same contract.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download

# Confirmed against vendor/microWakeWord/notebooks/basic_training_notebook.ipynb
# (cell 8) and corroborated live via the HF datasets API, which lists these
# exact filenames among kahrendt/microwakeword's siblings.
NEGATIVE_DATASET_REPO_ID = "kahrendt/microwakeword"
NEGATIVE_DATASET_FILENAMES = [
    "dinner_party.zip",
    "dinner_party_eval.zip",
    "no_speech.zip",
    "speech.zip",
]


def validate_dataset_dir(dir_path: Path) -> int:
    """Count Ragged Mmap feature folders found anywhere under ``dir_path``.

    microWakeWord's negative/background datasets are distributed as Ragged
    Mmap folders (directories whose name ends in "_mmap", written by
    mmap_ninja.ragged.RaggedMmap) rather than raw audio files. Its own
    ``MmapFeatureGenerator`` (vendor/microWakeWord/microwakeword/data.py)
    locates them the same way this function does: by globbing for
    directories ending in "_mmap".

    Raises:
        ValueError: if no such folders are found under ``dir_path``.
    """
    dir_path = Path(dir_path)
    count = sum(1 for p in dir_path.rglob("*_mmap") if p.is_dir())
    if count == 0:
        raise ValueError(f"No Ragged Mmap feature folders (*_mmap) found under {dir_path}")
    return count


def _download_and_extract(filename: str, out_dir: Path) -> None:
    """Download one dataset zip from the upstream HF dataset repo and extract it.

    Mirrors the notebook's `wget {link} && unzip {zip_path} -d {output_dir}`,
    using huggingface_hub + zipfile instead of shelling out.
    """
    zip_path = hf_hub_download(
        repo_id=NEGATIVE_DATASET_REPO_ID,
        repo_type="dataset",
        filename=filename,
    )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def main(
    argv: list[str] | None = None,
    download_and_extract: Callable[[str, Path], None] = _download_and_extract,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/negative_standard")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in NEGATIVE_DATASET_FILENAMES:
        download_and_extract(filename, out_dir)

    count = validate_dataset_dir(out_dir)
    print(f"Downloaded {count} Ragged Mmap feature folders to {out_dir}")


if __name__ == "__main__":
    main()
