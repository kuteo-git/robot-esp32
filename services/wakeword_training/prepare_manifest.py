"""Manifest preparation: deterministic train/val split of RAW WAV audio sources.

Scope (read this before adding a directory to --negative-dir or to
build_manifest()'s inputs):

    build_manifest() only spans directories of raw, un-processed ``.wav``
    files: ``data/positive`` (Task 5) and ``data/negative_vi/{hard,generic}``
    (Task 6). It deliberately does NOT include ``data/negative_standard``
    (Task 7's downloaded background/negative datasets).

Why negative_standard is excluded (researched against the vendored upstream
training pipeline, not assumed):

  * ``data/negative_standard`` is populated by
    ``download_background_datasets.py`` with the 4 zips
    (dinner_party, dinner_party_eval, no_speech, speech) that
    vendor/microWakeWord/notebooks/basic_training_notebook.ipynb cell 8
    downloads. Per that dataset's HF description ("This dataset contains
    spectrogram features in an mmap ninja format intended to use for
    microWakeWord training") and vendor/microWakeWord/microwakeword/data.py's
    ``MmapFeatureGenerator``, these are pre-extracted **Ragged Mmap** feature
    folders (dirs containing ``data/``, ``starts/``, ``ends/``, ``shapes/``,
    ``flattened_shapes/``, ``shapes_are_flat.ninja``) -- not raw audio. There
    are zero ``.wav`` files to glob; globbing them here would silently
    produce an empty split, which is exactly the bug this scope decision
    avoids.

  * Contrast with our own raw-audio sources: the notebook's own equivalent of
    "split raw audio into train/val" is the vendored ``Clips`` class
    (vendor/microWakeWord/microwakeword/audio/clips.py), which takes
    ``random_split_seed``/``split_count`` and internally calls HuggingFace
    ``datasets.train_test_split`` to produce train/test/validation subsets of
    raw clips (cell 5: ``Clips(input_directory='generated_samples', ...,
    random_split_seed=10, split_count=0.1)``). That's the same job
    build_manifest() does here (deterministic, seeded train/val split of raw
    audio *file paths*), just implemented directly instead of importing the
    heavier vendored/TensorFlow-dependent ``Clips`` machinery. The
    pre-extracted mmap folders have no raw-audio equivalent step to run --
    they arrive already split into their own ``training/validation/testing/
    testing_ambient`` subfolder structure (see ``MmapFeatureGenerator.dirs``
    and cell 9's comment: "Each feature_dir should have at least one of the
    following folders ... training/ validation/ testing/ testing_ambient/
    validation_ambient/"). Notably ``dinner_party_eval.zip`` is named as a
    distinct file from ``dinner_party.zip`` -- it already looks like a
    pre-made held-out/eval set upstream chose not to re-split, matching this
    same pattern (don't re-split what's already split).

  * The notebook DOES feature-extract raw audio (its TTS-generated positive
    samples) into Ragged Mmap format before training (cell 7: ``Clips`` +
    ``Augmentation`` + ``SpectrogramGeneration`` + ``RaggedMmap.from_generator``
    -> ``generated_augmented_features/{training,validation,testing}/
    wakeword_mmap``). That is real, but it is a full augmentation + spectrogram
    pipeline requiring TensorFlow, RIR/background-noise audio, and the
    vendored ``microwakeword.audio`` package -- a distinctly heavier and
    separate step from "manifest preparation" (splitting file paths). Task 9
    ("real training invocation") is where that feature-extraction step
    belongs: it will read this manifest's train/val file lists, run them
    through that same Clips/Augmentation/SpectrogramGeneration pipeline to
    produce its own Ragged Mmap folders (mirroring cell 7), and then build a
    training_parameters.yaml whose ``config["features"]`` list (cell 9)
    combines those newly-produced mmap folders together with the
    already-downloaded ``data/negative_standard`` mmap folders referenced
    directly by path -- exactly as cell 9 references
    ``negative_datasets/speech`` and ``negative_datasets/dinner_party``
    alongside ``generated_augmented_features``.

On the JSON shape write_manifest() produces: it intentionally does NOT try to
mimic microWakeWord's training_parameters.yaml ``config["features"]`` schema
(a list of ``{features_dir, sampling_weight, penalty_weight, truth,
truncation_strategy, type}`` dicts pointing at already-built Ragged Mmap
folders). This manifest is a strictly earlier-stage artifact -- a plain list
of raw file paths per split, analogous to what the vendored ``Clips`` class
holds internally after ``train_test_split`` (before any feature extraction
happens). No adaptation to line it up with the yaml config schema is needed
or appropriate: Task 9 converts this manifest's raw file lists into Ragged
Mmap folders first, and only *those* resulting folder paths are what get
plugged into the yaml config's ``features_dir`` entries.
"""
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
    """Split raw ``.wav`` sources into a deterministic train/val manifest.

    Only spans raw-audio directories (see module docstring for why
    ``data/negative_standard``'s pre-extracted Ragged Mmap folders must NOT
    be passed in here -- they'd silently contribute zero files).
    """
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
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic train/val manifest over raw-audio sources "
            "(data/positive, data/negative_vi/*). Does NOT include "
            "data/negative_standard: that directory holds pre-extracted "
            "Ragged Mmap feature folders (Task 7), not raw .wav files, and "
            "is referenced directly by Task 9's training config instead -- "
            "see this file's module docstring for the full reasoning."
        )
    )
    default_negative_dirs = [
        "data/negative_vi/hard",
        "data/negative_vi/generic",
        "data/negative_vi/chimes",       # robot's own chimes (fixes session-end loop)
        "data/negative_vi/robot_voice",  # robot's own VieNeu voice
        "data/negative_vi/vi_speech",    # real Vietnamese speech (FLEURS vi)
    ]
    parser.add_argument("--positive-dir", default="data/positive")
    parser.add_argument(
        "--negative-dir",
        action="append",
        default=None,
        help=(
            "Raw-.wav negative source directory; may be passed multiple "
            "times, replacing (not adding to) the default list below. "
            f"Defaults to the Task 6 Vietnamese negative dirs: "
            f"{default_negative_dirs}. Do NOT pass data/negative_standard "
            "here -- it holds pre-extracted Ragged Mmap folders (Task 7), "
            "not raw .wav files; Task 9 references it directly instead."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/manifest.json")
    args = parser.parse_args(argv)
    negative_dirs = args.negative_dir if args.negative_dir is not None else default_negative_dirs

    manifest = build_manifest(
        Path(args.positive_dir),
        [Path(d) for d in negative_dirs],
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
