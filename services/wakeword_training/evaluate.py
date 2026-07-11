"""Score a trained mai_oi.tflite model against a positive/negative WAV directory pair.

Works for both the synthetic held-out set (Task 8's val split) and the real-world set
captured via the Android app's mic test tool -- point --positive-dir/--negative-dir at
whichever set you want to evaluate.
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
