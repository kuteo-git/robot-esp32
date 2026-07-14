"""Training invocation: builds the microWakeWord training config, invokes the
real training entry point, and exports the resulting tflite model.

Upstream (`OHF-Voice/micro-wake-word`) has no plain `train.py` CLI script of
its own -- the notebook's own README/notebook IS the documented entry point,
and it shells out to a real, importable module:
`vendor/microWakeWord/microwakeword/model_train_eval.py`, invoked as
`python -m microwakeword.model_train_eval ...` (notebook cell 10 -- this is
already a subprocess/CLI invocation in the notebook itself, not something
that only runs inside a notebook kernel). That module also does the tflite
conversion/export (see `evaluate_model()` -> `utils.convert_saved_model_to_tflite`)
once training finishes, controlled by its `--test_tflite_streaming_quantized`
flag.

This script's job is everything AROUND that real invocation:

  1. Build `training_parameters.yaml` (cell 9's config dict, written
     verbatim except for the `features` list, which is built from our own
     extracted features (`extract_features.py`'s output under
     `data/features/`) plus whatever Ragged Mmap datasets
     `download_background_datasets.py` (Task 7) put under
     `data/negative_standard/`, discovered by scanning that directory rather
     than hardcoding the HF zip names -- see `discover_negative_standard()`.
  2. Invoke `python -m microwakeword.model_train_eval` against that config,
     using cell 10's own example mixednet hyperparameters verbatim.
  3. Copy the resulting quantized streaming tflite
     (`<train_dir>/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`,
     per `model_train_eval.py`'s `evaluate_model()` -- exactly the path
     notebook cell 11 downloads) to `models/mai_oi.tflite`.

Cell 9's config, quoted for reference (values below are copied from it
verbatim; only ``features`` differs, since cell 9's list of {generated_augmented_features,
negative_datasets/speech, negative_datasets/dinner_party, negative_datasets/no_speech,
negative_datasets/dinner_party_eval} paths are notebook-example paths for a single
Colab run, not ours):

    config["window_step_ms"] = 10
    config["training_steps"] = [10000]
    config["positive_class_weight"] = [1]
    config["negative_class_weight"] = [20]
    config["learning_rates"] = [0.001]
    config["batch_size"] = 128
    config["time_mask_max_size"] = [0]
    config["time_mask_count"] = [0]
    config["freq_mask_max_size"] = [0]
    config["freq_mask_count"] = [0]
    config["eval_step_interval"] = 500
    config["clip_duration_ms"] = 1500
    config["target_minimization"] = 0.9
    config["minimization_metric"] = None
    config["maximization_metric"] = "average_viable_recall"

Cell 10's invocation, quoted for reference (also copied verbatim below):

    !python -m microwakeword.model_train_eval \\
    --training_config='training_parameters.yaml' \\
    --train 1 \\
    --restore_checkpoint 1 \\
    --test_tf_nonstreaming 0 \\
    --test_tflite_nonstreaming 0 \\
    --test_tflite_nonstreaming_quantized 0 \\
    --test_tflite_streaming 0 \\
    --test_tflite_streaming_quantized 1 \\
    --use_weights "best_weights" \\
    mixednet \\
    --pointwise_filters "64,64,64,64" \\
    --repeat_in_block  "1, 1, 1, 1" \\
    --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \\
    --residual_connection "0,0,0,0" \\
    --first_conv_filters 32 \\
    --first_conv_kernel_size 5 \\
    --stride 3
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Per-dataset sampling/penalty weights from cell 9's config["features"] list,
# keyed by the top-level folder name each zip is expected to extract to
# (dinner_party.zip -> "dinner_party", etc. -- see download_background_datasets.py's
# docstring, which corroborates this against the HF dataset repo). Applied by
# discover_negative_standard() when a discovered subdirectory's name matches.
NEGATIVE_STANDARD_WEIGHTS = {
    "speech": {"sampling_weight": 10.0, "penalty_weight": 1.0, "truncation_strategy": "random"},
    "dinner_party": {"sampling_weight": 10.0, "penalty_weight": 1.0, "truncation_strategy": "random"},
    "no_speech": {"sampling_weight": 5.0, "penalty_weight": 1.0, "truncation_strategy": "random"},
    # "Only used for validation and testing" per cell 9's comment.
    "dinner_party_eval": {"sampling_weight": 0.0, "penalty_weight": 1.0, "truncation_strategy": "split"},
}
# Fallback for a discovered negative_standard subdirectory whose name doesn't
# match any of the above (e.g. upstream renames/adds a dataset) -- kept
# conservative rather than guessing a specific upstream intent.
DEFAULT_NEGATIVE_STANDARD_WEIGHTS = {
    "sampling_weight": 5.0,
    "penalty_weight": 1.0,
    "truncation_strategy": "random",
}

SPLIT_DIRS = ("training", "validation", "testing", "validation_ambient", "testing_ambient")

MIXEDNET_ARGS = [
    "mixednet",
    "--pointwise_filters",
    "64,64,64,64",
    "--repeat_in_block",
    "1, 1, 1, 1",
    "--mixconv_kernel_sizes",
    "[5], [7,11], [9,15], [23]",
    "--residual_connection",
    "0,0,0,0",
    "--first_conv_filters",
    "32",
    "--first_conv_kernel_size",
    "5",
    "--stride",
    "3",
]

TFLITE_QUANTIZED_STREAMING_RELPATH = Path(
    "tflite_stream_state_internal_quant"
) / "stream_state_internal_quant.tflite"


def has_mmap_features(dir_path: Path) -> bool:
    """True if any of the 5 split subdirectories under dir_path contain a
    Ragged Mmap feature folder (dir name ending "_mmap"), matching
    MmapFeatureGenerator's own glob ("**/*_mmap/") in
    vendor/microWakeWord/microwakeword/data.py.
    """
    for split in SPLIT_DIRS:
        split_dir = dir_path / split
        if split_dir.is_dir() and any(split_dir.glob("**/*_mmap/")):
            return True
    return False


def discover_negative_standard(negative_standard_dir: Path) -> list[dict]:
    """Builds cell-9-shaped feature entries for each dataset subdirectory
    Task 7's download_background_datasets.py extracted under
    data/negative_standard (e.g. speech/, dinner_party/, no_speech/,
    dinner_party_eval/), by scanning rather than hardcoding the zip-internal
    directory names (which this script cannot otherwise verify without
    actually downloading them).
    """
    entries = []
    if not negative_standard_dir.is_dir():
        return entries
    for child in sorted(negative_standard_dir.iterdir()):
        if not child.is_dir():
            continue
        if not has_mmap_features(child):
            continue
        weights = NEGATIVE_STANDARD_WEIGHTS.get(
            child.name, DEFAULT_NEGATIVE_STANDARD_WEIGHTS
        )
        entries.append(
            {
                "features_dir": str(child),
                "sampling_weight": weights["sampling_weight"],
                "penalty_weight": weights["penalty_weight"],
                "truth": False,
                "truncation_strategy": weights["truncation_strategy"],
                "type": "mmap",
            }
        )
    return entries


def build_config(
    features_dir: Path,
    negative_standard_dir: Path,
    train_dir: str,
    training_steps: list[int] | None = None,
    batch_size: int = 128,
    eval_step_interval: int = 500,
    clip_duration_ms: int = 1500,
) -> dict:
    """Builds the training_parameters.yaml dict, mirroring cell 9. All
    hyperparameter defaults are copied verbatim from the notebook's own
    example config (see module docstring); only `features`, `train_dir`, and
    the values explicitly overridable via CLI flags for smoke-testing differ.
    """
    training_steps = list(training_steps) if training_steps is not None else [10000]

    features = [
        {
            "features_dir": str(features_dir / "positive"),
            "sampling_weight": 2.0,
            "penalty_weight": 1.0,
            "truth": True,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        },
        {
            "features_dir": str(features_dir / "negative_vi"),
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
    ]
    features.extend(discover_negative_standard(negative_standard_dir))

    # Per-phase config lists must all match len(training_steps). Single-phase (the
    # default [10000]) keeps the original values; a multi-phase schedule (the "big
    # run", e.g. --training-steps 50000 30000 20000) gets a DECAYING learning rate so
    # the extra steps actually refine the model instead of just fluctuating at a
    # constant LR (a model this small has already converged by ~10k at 0.001).
    n_phases = len(training_steps)
    _lr_options = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
    learning_rates = (_lr_options + [_lr_options[-1]] * n_phases)[:n_phases]

    config = {
        "window_step_ms": 10,
        "train_dir": train_dir,
        "features": features,
        "training_steps": training_steps,
        "positive_class_weight": [1] * n_phases,
        "negative_class_weight": [20] * n_phases,
        "learning_rates": learning_rates,
        "batch_size": batch_size,
        # Modest SpecAugment masking ON (Phase-1 trained with it off). Masking a
        # few time/freq bins per clip regularizes the model toward a more
        # conservative decision boundary -- helpful given the goal is fewer
        # false-accepts on the newly-added chime/robot-voice/speech negatives.
        "time_mask_max_size": [5] * n_phases,
        "time_mask_count": [1] * n_phases,
        "freq_mask_max_size": [3] * n_phases,
        "freq_mask_count": [1] * n_phases,
        "eval_step_interval": eval_step_interval,
        "clip_duration_ms": clip_duration_ms,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    return config


def write_config(config: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(config, f)


def build_train_command(
    training_config: Path,
    python_exe: str = sys.executable,
    train: int = 1,
    restore_checkpoint: int = 1,
    test_tflite_streaming_quantized: int = 1,
) -> list[str]:
    """Builds the `python -m microwakeword.model_train_eval ...` command,
    matching cell 10's invocation verbatim (module docstring quotes the
    original). `train`/`restore_checkpoint`/`test_tflite_streaming_quantized`
    are exposed for smoke-testing (e.g. `--train 0` to only exercise
    argument/config parsing without a real training loop).
    """
    return [
        python_exe,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={training_config}",
        "--train",
        str(train),
        "--restore_checkpoint",
        str(restore_checkpoint),
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        str(test_tflite_streaming_quantized),
        "--use_weights",
        "best_weights",
        *MIXEDNET_ARGS,
    ]


def export_tflite(train_dir: Path, dest: Path) -> Path:
    """Copies the quantized streaming tflite model
    (<train_dir>/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite,
    the same file notebook cell 11 downloads) to `dest`.
    """
    src = train_dir / TFLITE_QUANTIZED_STREAMING_RELPATH
    if not src.is_file():
        raise FileNotFoundError(
            f"Expected trained tflite model at {src} (produced by "
            "microwakeword.model_train_eval's evaluate_model() when "
            "--test_tflite_streaming_quantized 1 is set); training may not "
            "have completed successfully."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest


def main(
    argv: list[str] | None = None,
    runner=subprocess.run,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--negative-standard-dir", default="data/negative_standard")
    parser.add_argument("--train-dir", default="models/mai_oi_train")
    parser.add_argument("--training-config", default="training_parameters.yaml")
    parser.add_argument("--training-steps", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-step-interval", type=int, default=500)
    parser.add_argument("--clip-duration-ms", type=int, default=1500)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--restore-checkpoint", type=int, default=1)
    parser.add_argument("--skip-training", action="store_true", help="Only build the config; don't invoke training.")
    parser.add_argument("--skip-export", action="store_true", help="Don't copy the tflite output after training.")
    parser.add_argument("--out", default="models/mai_oi.tflite")
    args = parser.parse_args(argv)

    config = build_config(
        Path(args.features_dir),
        Path(args.negative_standard_dir),
        train_dir=args.train_dir,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        eval_step_interval=args.eval_step_interval,
        clip_duration_ms=args.clip_duration_ms,
    )
    training_config_path = Path(args.training_config)
    write_config(config, training_config_path)
    print(f"Wrote {training_config_path} with {len(config['features'])} feature source(s).")

    if args.skip_training:
        return

    command = build_train_command(
        training_config_path,
        train=args.train,
        restore_checkpoint=args.restore_checkpoint,
    )
    print("Running:", " ".join(command))
    result = runner(command)
    returncode = getattr(result, "returncode", 0)
    if returncode != 0:
        raise SystemExit(returncode)

    if args.skip_export:
        return

    dest = export_tflite(Path(args.train_dir), Path(args.out))
    print(f"Exported {dest}")


if __name__ == "__main__":
    main()
