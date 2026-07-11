"""Score a trained mai_oi.tflite model against a positive/negative set of WAV files.

Two mutually-exclusive input modes (pick exactly one):

  1. Manifest mode (``--manifest``/``--split``) -- for the synthetic held-out set
     produced by Task 8's ``prepare_manifest.py``. That script writes
     ``data/manifest.json`` as a JSON object of
     ``{"train"/"val": {"positive"/"negative": [<raw WAV file path>, ...]}}`` --
     individual file paths, never materialized as a directory of WAVs. Pass
     ``--manifest data/manifest.json --split val`` to score exactly the file paths
     listed under ``manifest["val"]["positive"]``/``manifest["val"]["negative"]``
     directly, with no directory-materialization step required.

  2. Directory mode (``--positive-dir``/``--negative-dir``) -- for the real-world set
     captured via the Android app's mic test tool (``MicTest.kt``), which really is
     saved to disk as a directory of WAV files. Every ``*.wav`` file directly inside
     each directory is globbed and scored.

Do not mix the two modes (e.g. --manifest with --positive-dir): supply either
--manifest/--split, or both --positive-dir and --negative-dir.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Callable

from metrics import compute_metrics

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "vendor" / "microWakeWord")
)

# Matches extract_features.py's STEP_MS / train.py's build_config()
# config["window_step_ms"] -- the spectrogram window step (in ms) used both
# when extracting training features and when the real trained model was
# fitted. Evaluation must use the same step size so the spectrogram frames
# fed to the interpreter line up with what the model was trained on.
WINDOW_STEP_MS = 10

# Matches train.py's MIXEDNET_ARGS ("--stride", "3"), which becomes
# config["stride"] in vendor/microWakeWord/microwakeword/model_train_eval.py's
# load_config() (`config["stride"] = flags.__dict__.get("stride", 1)`) --
# the same value vendor/microWakeWord/microwakeword/test.py's
# tflite_streaming_model_roc() passes to `Model(..., stride=config["stride"])`
# when it computes its own per-clip streaming scores for the ROC/FRR-FAR
# report. Our mai_oi.tflite is always built with that fixed mixednet
# architecture, so this is hardcoded here (evaluate.py's CLI has no
# training_parameters.yaml to read config["stride"] from) rather than
# threaded through as a flag.
MIXEDNET_STRIDE = 3


class TFLiteScorer:
    """Runs the real streaming-inference contract from
    vendor/microWakeWord/microwakeword/inference.py's ``Model`` class against
    a WAV file and reduces its per-chunk streaming probabilities to a single
    float score.

    ``Model`` already implements exactly the input-tensor-shape/streaming
    handling this needs (see inference.py's ``predict_clip``/
    ``predict_spectrogram``): it reads the model's own input tensor time
    dimension (``input_details[0]["shape"][1]``) rather than a guessed
    constant, slices the spectrogram into that many frames per chunk,
    strided by ``MIXEDNET_STRIDE`` frames, quantizes/dequantizes as needed
    for the int8 model produced by train.py's
    ``--test_tflite_streaming_quantized 1``, and relies on the tflite
    interpreter's own internal streaming-state variables (fed back
    automatically between ``invoke()`` calls by the "stream_state_internal"
    graph) rather than any manual state array this code would have to
    manage -- mirroring vendor/microWakeWord/microwakeword/test.py's
    ``tflite_streaming_model_roc``, which is the same function
    train.py/evaluate_model() rely on for the model's own FRR/FAR reporting.

    CAVEAT -- per-clip aggregation is a SIMPLIFIED variant of upstream's own
    ROC methodology, not a reproduction of it. ``tflite_streaming_model_roc()``
    (vendor/microWakeWord/microwakeword/test.py, ~lines 367-373) scores a
    positive clip by first dropping the first ``ignore_slices_after_accept``
    (25) chunks, then taking a 5-frame moving average
    (``sliding_window_view(..., sliding_window_length=5).mean(axis=-1)``)
    over what remains, and only THEN taking ``np.max`` of that
    trimmed+smoothed sequence. ``score_wav_file`` below skips both the
    warm-up trim and the smoothing and takes a plain ``max()`` over the raw
    per-chunk probabilities instead (see that method's docstring for why).
    Consequence: this class's scores, and therefore evaluate.py's FRR/FAR
    report, are NOT directly comparable to the numbers upstream's own
    ``model_train_eval.py --test_tflite_streaming_quantized`` run would
    report for the identical model -- unsmoothed raw-max can be
    systematically more optimistic on the negative set, since a single
    transient probability spike near the threshold is never smoothed away
    here the way upstream's moving average would suppress it. Treat this
    report's FRR/FAR as this project's own evaluation metric, not as a
    stand-in for upstream's ROC metric.
    """

    def __init__(self, model_path: str):
        from microwakeword.inference import Model

        self.model_path = model_path
        self._model = Model(model_path, stride=MIXEDNET_STRIDE)

    def score_wav_file(self, wav_path: Path) -> float:
        import soundfile as sf

        audio, sample_rate = sf.read(str(wav_path), dtype="int16")
        if sample_rate != 16000:
            raise ValueError(
                f"{wav_path}: expected 16kHz audio (matching the model's "
                f"training data), got {sample_rate}Hz"
            )

        probabilities = self._model.predict_clip(audio, step_ms=WINDOW_STEP_MS)
        if not probabilities:
            # Clip too short to produce even one full spectrogram chunk.
            return 0.0

        # Per-clip score = max probability seen at any point in the stream
        # (a wake word can occur anywhere in the clip, not just at a fixed
        # position), which is the same core idea as
        # vendor/microWakeWord/microwakeword/test.py's
        # tflite_streaming_model_roc() (each positive clip's score there is
        # also an `np.max(...)` over its streaming probability sequence).
        #
        # NOT the same numbers, though: upstream first drops the first 25
        # chunks (`ignore_slices_after_accept`) and applies a 5-frame moving
        # average before taking that max (see the class docstring's CAVEAT
        # section). This method deliberately skips both steps and maxes the
        # raw per-chunk probabilities directly -- simpler, but the resulting
        # FRR/FAR is this project's own metric, not directly comparable to
        # upstream's own training-time ROC evaluation of the same model.
        return float(max(probabilities))


def score_paths(scorer, paths: list[str]) -> list[float]:
    """Scores an explicit list of WAV file paths, in the given order."""
    return [scorer.score_wav_file(Path(p)) for p in paths]


def score_directory(scorer, dir_path: Path) -> list[float]:
    """Scores every ``*.wav`` file directly inside ``dir_path`` (sorted for
    determinism). Thin wrapper around ``score_paths`` for the directory-based
    real-world-eval mode."""
    return score_paths(scorer, sorted(str(p) for p in Path(dir_path).glob("*.wav")))


def _load_manifest_split(manifest_path: str, split: str) -> tuple[list[str], list[str]]:
    """Reads Task 8's ``prepare_manifest.py`` manifest.json and returns the
    ``(positive_paths, negative_paths)`` file-path lists for ``split``
    ("train" or "val")."""
    manifest = json.loads(Path(manifest_path).read_text())
    if split not in manifest:
        raise ValueError(
            f"{manifest_path}: no {split!r} split in manifest "
            f"(available: {sorted(manifest.keys())})"
        )
    split_data = manifest[split]
    return list(split_data.get("positive", [])), list(split_data.get("negative", []))


def main(
    argv: list[str] | None = None,
    scorer_factory: Callable[[str], object] = TFLiteScorer,
) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--manifest",
        help="Path to manifest.json (Task 8's prepare_manifest.py output). "
        "Requires --split. Mutually exclusive with --positive-dir/--negative-dir.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        help="Which manifest split to score ('train' or 'val'). Requires --manifest.",
    )
    parser.add_argument(
        "--positive-dir",
        help="Directory of positive *.wav files to glob and score. Requires "
        "--negative-dir. Mutually exclusive with --manifest/--split.",
    )
    parser.add_argument(
        "--negative-dir",
        help="Directory of negative *.wav files to glob and score. Requires "
        "--positive-dir. Mutually exclusive with --manifest/--split.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args(argv)

    manifest_mode = args.manifest is not None or args.split is not None
    directory_mode = args.positive_dir is not None or args.negative_dir is not None

    if manifest_mode and directory_mode:
        parser.error(
            "--manifest/--split and --positive-dir/--negative-dir are mutually "
            "exclusive -- pick one mode."
        )
    if not manifest_mode and not directory_mode:
        parser.error(
            "must supply either --manifest and --split, or both --positive-dir "
            "and --negative-dir."
        )
    if manifest_mode and (args.manifest is None or args.split is None):
        parser.error("--manifest and --split must be supplied together.")
    if directory_mode and (args.positive_dir is None or args.negative_dir is None):
        parser.error("--positive-dir and --negative-dir must be supplied together.")

    scorer = scorer_factory(args.model)
    if manifest_mode:
        positive_paths, negative_paths = _load_manifest_split(args.manifest, args.split)
        positive_scores = score_paths(scorer, positive_paths)
        negative_scores = score_paths(scorer, negative_paths)
    else:
        positive_scores = score_directory(scorer, Path(args.positive_dir))
        negative_scores = score_directory(scorer, Path(args.negative_dir))

    result = compute_metrics(positive_scores, negative_scores, args.threshold)

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dataclasses.asdict(result), indent=2))
    print(
        f"FRR={result.false_reject_rate:.3f} "
        f"FAR={result.false_accept_rate:.3f} "
        f"({result.num_positive} positive, {result.num_negative} negative)"
    )


if __name__ == "__main__":
    main()
