"""Tests for train.py: training config construction, negative_standard
discovery, the exact training-invocation command, and tflite export.

test_main_invokes_model_train_eval_with_expected_command_and_exports is the
closest thing to an end-to-end smoke test that doesn't actually run
TensorFlow training: it injects a fake `runner` (mirroring
download_background_datasets.py's `download_and_extract` injection pattern)
that captures the exact command train.py would execute and plants a fake
tflite output, then asserts main() copies it to the requested --out path.
The *real* `python -m microwakeword.model_train_eval` invocation (that
exact command, unmodified) was run manually against a tiny real extracted
feature set during Task 9's development and confirmed to build the model,
train for a couple of steps, and export a real quantized streaming tflite
file end-to-end -- see the Task 9 report for that transcript. It is not
re-run here because a real training subprocess call would make this test
suite slow and TensorFlow-environment-dependent.
"""
from pathlib import Path

import pytest
import yaml

from train import (
    MIXEDNET_ARGS,
    build_config,
    build_train_command,
    discover_negative_standard,
    export_tflite,
    has_mmap_features,
    main,
)


def _make_mmap_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "shapes_are_flat.ninja").write_text("1")


def test_has_mmap_features_true_when_any_split_has_mmap_folder(tmp_path):
    _make_mmap_dir(tmp_path / "training" / "foo_mmap")
    assert has_mmap_features(tmp_path) is True


def test_has_mmap_features_false_for_empty_or_non_mmap_dir(tmp_path):
    (tmp_path / "training" / "not_a_feature_dir").mkdir(parents=True)
    assert has_mmap_features(tmp_path) is False
    assert has_mmap_features(tmp_path / "does_not_exist") is False


def test_discover_negative_standard_maps_known_dataset_weights(tmp_path):
    _make_mmap_dir(tmp_path / "speech" / "training" / "a_mmap")
    _make_mmap_dir(tmp_path / "dinner_party" / "training" / "b_mmap")
    _make_mmap_dir(tmp_path / "no_speech" / "training" / "c_mmap")
    _make_mmap_dir(tmp_path / "dinner_party_eval" / "testing" / "d_mmap")

    entries = discover_negative_standard(tmp_path)
    by_name = {Path(e["features_dir"]).name: e for e in entries}

    assert len(entries) == 4
    assert by_name["speech"]["sampling_weight"] == 10.0
    assert by_name["dinner_party"]["sampling_weight"] == 10.0
    assert by_name["no_speech"]["sampling_weight"] == 5.0
    # "Only used for validation and testing" per notebook cell 9's comment.
    assert by_name["dinner_party_eval"]["sampling_weight"] == 0.0
    assert by_name["dinner_party_eval"]["truncation_strategy"] == "split"
    assert all(e["truth"] is False for e in entries)
    assert all(e["type"] == "mmap" for e in entries)


def test_discover_negative_standard_uses_fallback_weights_for_unknown_dataset(tmp_path):
    _make_mmap_dir(tmp_path / "some_future_dataset" / "training" / "x_mmap")
    entries = discover_negative_standard(tmp_path)
    assert len(entries) == 1
    assert entries[0]["sampling_weight"] == 5.0
    assert entries[0]["truncation_strategy"] == "random"


def test_discover_negative_standard_skips_dirs_without_mmap_features(tmp_path):
    (tmp_path / "README" ).mkdir()
    (tmp_path / "not_a_dataset").mkdir()
    (tmp_path / "not_a_dataset" / "readme.txt").write_text("x")
    assert discover_negative_standard(tmp_path) == []


def test_discover_negative_standard_returns_empty_for_missing_dir(tmp_path):
    assert discover_negative_standard(tmp_path / "does_not_exist") == []


def test_build_config_matches_notebook_cell_9_defaults(tmp_path):
    config = build_config(
        features_dir=tmp_path / "features",
        negative_standard_dir=tmp_path / "does_not_exist",
        train_dir="trained_models/wakeword",
    )
    # Hyperparameters copied verbatim from cell 9 -- see train.py's module
    # docstring for the quoted source. Only `features`/`train_dir` differ
    # from the notebook's own example values.
    assert config["window_step_ms"] == 10
    assert config["training_steps"] == [10000]
    assert config["positive_class_weight"] == [1]
    assert config["negative_class_weight"] == [20]
    assert config["learning_rates"] == [0.001]
    assert config["batch_size"] == 128
    assert config["time_mask_max_size"] == [0]
    assert config["time_mask_count"] == [0]
    assert config["freq_mask_max_size"] == [0]
    assert config["freq_mask_count"] == [0]
    assert config["eval_step_interval"] == 500
    assert config["clip_duration_ms"] == 1500
    assert config["target_minimization"] == 0.9
    assert config["minimization_metric"] is None
    assert config["maximization_metric"] == "average_viable_recall"

    positive_entry, negative_vi_entry = config["features"]
    assert positive_entry["truth"] is True
    assert positive_entry["truncation_strategy"] == "truncate_start"
    assert positive_entry["sampling_weight"] == 2.0
    assert negative_vi_entry["truth"] is False
    assert negative_vi_entry["truncation_strategy"] == "random"
    assert negative_vi_entry["sampling_weight"] == 10.0


def test_build_config_combines_own_features_with_negative_standard(tmp_path):
    _make_mmap_dir(tmp_path / "negative_standard" / "speech" / "training" / "s_mmap")

    config = build_config(
        features_dir=tmp_path / "features",
        negative_standard_dir=tmp_path / "negative_standard",
        train_dir="trained_models/wakeword",
    )
    # positive + negative_vi (ours) + speech (negative_standard) = 3.
    assert len(config["features"]) == 3
    dirs = [Path(f["features_dir"]).name for f in config["features"]]
    assert dirs == ["positive", "negative_vi", "speech"]


def test_build_config_is_yaml_serializable(tmp_path):
    config = build_config(
        features_dir=tmp_path / "features",
        negative_standard_dir=tmp_path / "does_not_exist",
        train_dir="trained_models/wakeword",
    )
    dumped = yaml.dump(config)
    reloaded = yaml.safe_load(dumped)
    assert reloaded == config


def test_build_train_command_matches_notebook_cell_10():
    command = build_train_command(Path("training_parameters.yaml"), python_exe="python")

    assert command[:3] == ["python", "-m", "microwakeword.model_train_eval"]
    assert "--training_config=training_parameters.yaml" in command
    assert "--train" in command and command[command.index("--train") + 1] == "1"
    assert "--restore_checkpoint" in command
    assert command[command.index("--restore_checkpoint") + 1] == "1"
    assert "--test_tflite_streaming_quantized" in command
    assert command[command.index("--test_tflite_streaming_quantized") + 1] == "1"
    # All other test_* flags are disabled (0), matching cell 10 verbatim.
    for flag in (
        "--test_tf_nonstreaming",
        "--test_tflite_nonstreaming",
        "--test_tflite_nonstreaming_quantized",
        "--test_tflite_streaming",
    ):
        assert command[command.index(flag) + 1] == "0"
    assert command[-len(MIXEDNET_ARGS):] == MIXEDNET_ARGS
    assert "--pointwise_filters" in MIXEDNET_ARGS


def test_build_train_command_train_0_for_eval_only():
    command = build_train_command(Path("cfg.yaml"), train=0, python_exe="python")
    assert command[command.index("--train") + 1] == "0"


def test_export_tflite_copies_expected_quantized_streaming_model(tmp_path):
    train_dir = tmp_path / "trained_models" / "wakeword"
    tflite_path = train_dir / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    tflite_path.parent.mkdir(parents=True)
    tflite_path.write_bytes(b"fake-tflite-bytes")

    dest = export_tflite(train_dir, tmp_path / "models" / "mai_oi.tflite")

    assert dest.is_file()
    assert dest.read_bytes() == b"fake-tflite-bytes"


def test_export_tflite_raises_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="tflite_stream_state_internal_quant"):
        export_tflite(tmp_path / "trained_models" / "wakeword", tmp_path / "out.tflite")


def test_main_invokes_model_train_eval_with_expected_command_and_exports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured_commands = []

    train_dir = tmp_path / "models" / "mai_oi_train"
    tflite_path = train_dir / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"

    def fake_runner(command):
        captured_commands.append(command)
        # Simulate microwakeword.model_train_eval's real export step (see
        # model_train_eval.py's evaluate_model()) by planting a file at the
        # same relative path it would produce.
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        tflite_path.write_bytes(b"fake-tflite-bytes")

        class Result:
            returncode = 0

        return Result()

    main(
        [
            "--features-dir",
            str(tmp_path / "data" / "features"),
            "--negative-standard-dir",
            str(tmp_path / "data" / "negative_standard"),
            "--train-dir",
            str(train_dir),
            "--training-config",
            str(tmp_path / "training_parameters.yaml"),
            "--training-steps",
            "2",
            "--batch-size",
            "2",
            "--out",
            str(tmp_path / "models" / "mai_oi.tflite"),
        ],
        runner=fake_runner,
    )

    assert len(captured_commands) == 1
    command = captured_commands[0]
    assert command[1:3] == ["-m", "microwakeword.model_train_eval"]
    assert f"--training_config={tmp_path / 'training_parameters.yaml'}" in command

    written_config = yaml.safe_load((tmp_path / "training_parameters.yaml").read_text())
    assert written_config["training_steps"] == [2]
    assert written_config["batch_size"] == 2

    out_path = tmp_path / "models" / "mai_oi.tflite"
    assert out_path.is_file()
    assert out_path.read_bytes() == b"fake-tflite-bytes"


def test_main_raises_on_nonzero_training_returncode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def failing_runner(command):
        class Result:
            returncode = 1

        return Result()

    with pytest.raises(SystemExit):
        main(
            [
                "--training-config",
                str(tmp_path / "training_parameters.yaml"),
                "--train-dir",
                str(tmp_path / "models" / "mai_oi_train"),
            ],
            runner=failing_runner,
        )


def test_main_skip_training_only_writes_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def unexpected_runner(command):
        calls.append(command)
        raise AssertionError("training runner should not be invoked with --skip-training")

    main(
        [
            "--training-config",
            str(tmp_path / "training_parameters.yaml"),
            "--skip-training",
        ],
        runner=unexpected_runner,
    )

    assert calls == []
    assert (tmp_path / "training_parameters.yaml").is_file()
