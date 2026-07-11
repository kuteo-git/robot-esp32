"""Feature extraction: raw WAV manifest -> Ragged Mmap spectrogram features.

This is the missing step surfaced by Task 9's research (see the module
docstring of ``prepare_manifest.py`` and
``vendor/microWakeWord/notebooks/basic_training_notebook.ipynb``): training
consumes pre-extracted spectrogram features stored as Ragged Mmap folders,
not raw WAV files. This script produces those folders for OUR raw audio
(``data/positive`` and ``data/negative_vi/{hard,generic}``, combined by
``prepare_manifest.py`` into ``data/manifest.json``), mirroring notebook
cells 5-7 as closely as possible while reusing that manifest's already
deterministic split instead of re-deriving a second, different one.

Notebook cells this mirrors (vendor/microWakeWord/notebooks/basic_training_notebook.ipynb):

  Cell 5 ("Sets up the augmentations"):

      clips = Clips(input_directory='generated_samples', file_pattern='*.wav',
                     max_clip_duration_s=None, remove_silence=False,
                     random_split_seed=10, split_count=0.1)
      augmenter = Augmentation(augmentation_duration_s=3.2,
                     augmentation_probabilities={... "AddBackgroundNoise": 0.75,
                     "Gain": 1.0, "RIR": 0.5, ...},
                     impulse_paths=['mit_rirs'],
                     background_paths=['fma_16k', 'audioset_16k'],
                     background_min_snr_db=-5, background_max_snr_db=10,
                     min_jitter_s=0.195, max_jitter_s=0.205)

  Cell 7 ("Augment samples and save the training, validation, and testing
  sets"): for each of the "training"/"validation"/"testing" splits, builds a
  ``SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=...,
  step_ms=10)`` (slide_frames=10 + repetition=2 for "training" to simulate
  streaming inference; slide_frames=10 + repetition=1 for "validation";
  slide_frames=1 + repetition=1, no artificial repeat, for "testing") and
  writes it via ``RaggedMmap.from_generator(out_dir=..., sample_generator=
  spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
  batch_size=100)``.

Why ``Clips`` isn't reused directly: ``Clips.__init__`` only knows how to
load a whole directory (``input_directory`` + ``file_pattern`` glob) and
then, if given a ``random_split_seed``, re-split it *itself* via
``datasets.Dataset.train_test_split``. Task 8's ``prepare_manifest.py``
already produced a deterministic train/val split of the same source
directories and recorded it in ``data/manifest.json`` -- calling ``Clips``
with its own ``random_split_seed`` over those directories would silently
produce a SECOND, DIFFERENT split, disagreeing with the recorded manifest.
``ManifestClips`` below duck-types the exact interface
``SpectrogramGeneration`` needs from a "clips" object
(``audio_generator(split=None, repeat=1)`` / ``get_random_clip()``, see
``microwakeword/audio/clips.py``) over an explicit path list instead,
reusing the identical ``datasets.Audio(sampling_rate=16000)`` decode
``Clips`` uses internally so the resulting spectrograms are produced by the
exact same code path.

Known gap (flagged, not hidden): the notebook's cell 4 downloads MIT RIR
impulse responses and AudioSet/FMA background noise clips to feed
``Augmentation``'s ``impulse_paths``/``background_paths``. Tasks 1-8 of this
plan never built an equivalent download step, so by default this script
passes empty ``impulse_paths``/``background_paths`` (RIR and
AddBackgroundNoise become no-ops per ``Augmentation``'s own identity-transform
fallback when a path list is empty -- see
``microwakeword/audio/augmentation.py``). ``--rir-dir`` / ``--background-dir``
are provided so a real run can supply real directories of impulse-response /
background-noise WAVs if/when that data is fetched; until then, extracted
features rely only on the other augmentations (EQ, distortion, pitch shift,
band-stop, color noise, gain) plus whatever intrinsic noise Task 6's
generated Vietnamese negatives and Task 5's TTS positives already contain.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import datasets

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "vendor" / "microWakeWord")
)

from mmap_ninja.ragged import RaggedMmap  # noqa: E402

from microwakeword.audio.augmentation import Augmentation  # noqa: E402
from microwakeword.audio.spectrograms import SpectrogramGeneration  # noqa: E402

# Verbatim from cell 5 -- upstream's own example augmentation settings, not
# invented here.
AUGMENTATION_PROBABILITIES = {
    "SevenBandParametricEQ": 0.1,
    "TanhDistortion": 0.1,
    "PitchShift": 0.1,
    "BandStopFilter": 0.1,
    "AddColorNoise": 0.1,
    "AddBackgroundNoise": 0.75,
    "Gain": 1.0,
    "RIR": 0.5,
}

# (output split name) -> (manifest split key, repetition, slide_frames).
# Mirrors cell 7's three branches. data/manifest.json (Task 8) only has
# "train"/"val" splits (no held-out "test"), so -- documented tradeoff, see
# module docstring -- the "val" clips are reused for both the "validation"
# and "testing" outputs, each processed with cell 7's respective
# slide_frames/repetition settings for that split. A dedicated third split
# would require extending prepare_manifest.py, which is out of this task's
# scope (that file belongs to Task 8).
SPLIT_PLAN = {
    "training": ("train", 2, 10),
    "validation": ("val", 1, 10),
    "testing": ("val", 1, 1),
}

STEP_MS = 10  # Matches cell 7's SpectrogramGeneration(..., step_ms=10) and
# cell 9's config["window_step_ms"] = 10.


class ManifestClips:
    """Duck-types ``Clips``' generator interface over an explicit path list.

    See the module docstring for why this exists instead of using the real
    ``Clips`` class's own directory-glob + internal split.
    """

    def __init__(self, paths: list[str]):
        if paths:
            dataset = datasets.Dataset.from_dict({"audio": list(paths)}).cast_column(
                "audio", datasets.Audio(sampling_rate=16000)
            )
        else:
            dataset = []
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def audio_generator(self, split: str | None = None, repeat: int = 1):
        # `split` is accepted only for call-signature compatibility with
        # SpectrogramGeneration.spectrogram_generator, which always passes
        # split=<name> (cell 7). This instance already holds exactly one
        # (source, split) pair's clips, so there is nothing left to select.
        del split
        for _ in range(repeat):
            for clip in self._dataset:
                yield clip["audio"]["array"]

    def get_random_clip(self):
        return random.choice(self._dataset)["audio"]["array"]


def build_augmenter(
    rir_dirs: list[str] | None = None, background_dirs: list[str] | None = None
) -> Augmentation:
    """Builds the notebook cell 5 ``Augmentation`` object, upstream's example config."""
    return Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities=AUGMENTATION_PROBABILITIES,
        impulse_paths=list(rir_dirs or []),
        background_paths=list(background_dirs or []),
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def extract_source(
    manifest: dict,
    manifest_key: str,
    out_dir: Path,
    mmap_name: str,
    augmenter: Augmentation,
) -> dict[str, int]:
    """Feature-extracts one manifest key (e.g. "positive" or "negative") into
    ``out_dir/{training,validation,testing}/<mmap_name>_mmap`` Ragged Mmap
    folders, mirroring cell 7. Returns a dict of split -> spectrogram count
    written, for logging/testing.

    Splits with zero source clips are skipped (RaggedMmap.from_generator over
    zero samples would otherwise write a degenerate/empty mmap folder that
    ``MmapFeatureGenerator`` would then have to special-case).
    """
    counts: dict[str, int] = {}
    for out_split, (manifest_split, repeat, slide_frames) in SPLIT_PLAN.items():
        paths = manifest.get(manifest_split, {}).get(manifest_key, [])
        if not paths:
            counts[out_split] = 0
            continue

        clips = ManifestClips(paths)
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=STEP_MS,
        )

        mmap_out_dir = out_dir / out_split / f"{mmap_name}_mmap"
        # mmap_ninja's from_generator_base does out_dir.mkdir(exist_ok=True)
        # -- NOT parents=True -- so the parent chain must already exist.
        mmap_out_dir.parent.mkdir(parents=True, exist_ok=True)
        # RaggedMmap.from_generator returns the resulting RaggedMmap (or None
        # if the generator produced zero samples), NOT an iterator over
        # samples -- see mmap_ninja.base.from_generator_base.
        written_mmap = RaggedMmap.from_generator(
            out_dir=str(mmap_out_dir),
            sample_generator=spectrograms.spectrogram_generator(
                split=out_split, repeat=repeat
            ),
            batch_size=100,
            verbose=True,
        )
        counts[out_split] = len(written_mmap) if written_mmap is not None else 0
    return counts


def extract_all(
    manifest: dict,
    out_dir: Path,
    rir_dirs: list[str] | None = None,
    background_dirs: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Extracts both the positive ("data/positive") and negative
    ("data/negative_vi/*", combined by the manifest under the "negative" key)
    manifest classes into separate ``features_dir``-shaped folders.
    """
    augmenter = build_augmenter(rir_dirs, background_dirs)
    results = {}
    results["positive"] = extract_source(
        manifest, "positive", out_dir / "positive", "positive", augmenter
    )
    results["negative_vi"] = extract_source(
        manifest, "negative", out_dir / "negative_vi", "negative_vi", augmenter
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--out-dir", default="data/features")
    parser.add_argument(
        "--rir-dir",
        action="append",
        default=[],
        help="Directory of room-impulse-response WAVs for the RIR augmentation "
        "(cell 5's 'mit_rirs'). Optional -- omit to disable RIR augmentation. "
        "May be passed multiple times.",
    )
    parser.add_argument(
        "--background-dir",
        action="append",
        default=[],
        help="Directory of background-noise WAVs for the AddBackgroundNoise "
        "augmentation (cell 5's 'fma_16k'/'audioset_16k'). Optional -- omit to "
        "disable background-noise augmentation. May be passed multiple times.",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    results = extract_all(
        manifest,
        Path(args.out_dir),
        rir_dirs=args.rir_dir,
        background_dirs=args.background_dir,
    )
    for name, counts in results.items():
        print(f"{name}: {counts}")


if __name__ == "__main__":
    main()
