import json

import numpy as np
import pytest
import soundfile as sf

import evaluate


class _FakeScorer:
    def __init__(self, model_path):
        self.model_path = model_path

    def score_wav_file(self, wav_path):
        # Deterministic fake score based on filename so the test can assert exact values.
        return 0.9 if "pos" in str(wav_path) else 0.1


def test_score_directory_scores_every_wav(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.wav").write_bytes(b"")
    scores = evaluate.score_directory(_FakeScorer("unused"), tmp_path)
    assert len(scores) == 2


def test_main_writes_report_json(tmp_path):
    pos_dir = tmp_path / "positive"
    neg_dir = tmp_path / "negative"
    pos_dir.mkdir()
    neg_dir.mkdir()
    sf.write(str(pos_dir / "pos_1.wav"), np.zeros(100, dtype=np.float32), 16000)
    sf.write(str(neg_dir / "neg_1.wav"), np.zeros(100, dtype=np.float32), 16000)

    report_path = tmp_path / "report.json"
    evaluate.main(
        argv=[
            "--model", "unused.tflite",
            "--positive-dir", str(pos_dir),
            "--negative-dir", str(neg_dir),
            "--threshold", "0.5",
            "--report-out", str(report_path),
        ],
        scorer_factory=lambda model_path: _FakeScorer(model_path),
    )

    report = json.loads(report_path.read_text())
    assert report["num_positive"] == 1
    assert report["num_negative"] == 1
    assert report["false_reject_rate"] == 0.0
    assert report["false_accept_rate"] == 0.0


def test_main_creates_report_out_parent_directory_if_missing(tmp_path):
    pos_dir = tmp_path / "positive"
    neg_dir = tmp_path / "negative"
    pos_dir.mkdir()
    neg_dir.mkdir()
    sf.write(str(pos_dir / "pos_1.wav"), np.zeros(100, dtype=np.float32), 16000)
    sf.write(str(neg_dir / "neg_1.wav"), np.zeros(100, dtype=np.float32), 16000)

    # reports/ does not exist yet under tmp_path.
    report_path = tmp_path / "reports" / "nested" / "report.json"
    evaluate.main(
        argv=[
            "--model", "unused.tflite",
            "--positive-dir", str(pos_dir),
            "--negative-dir", str(neg_dir),
            "--threshold", "0.5",
            "--report-out", str(report_path),
        ],
        scorer_factory=lambda model_path: _FakeScorer(model_path),
    )

    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["num_positive"] == 1


def test_score_paths_scores_every_path(tmp_path):
    (tmp_path / "pos_1.wav").write_bytes(b"")
    (tmp_path / "neg_1.wav").write_bytes(b"")
    paths = [str(tmp_path / "pos_1.wav"), str(tmp_path / "neg_1.wav")]
    scores = evaluate.score_paths(_FakeScorer("unused"), paths)
    assert scores == [0.9, 0.1]


def test_main_with_manifest_split_scores_exactly_those_paths(tmp_path):
    # Task 8's prepare_manifest.py never materializes val-split file lists as
    # directories -- it writes a JSON manifest of individual file paths per
    # split. This test builds a small fake manifest.json (matching that real
    # on-disk shape) and asserts --manifest/--split reads it directly, rather
    # than requiring a --positive-dir/--negative-dir pair of directories that
    # nothing in the pipeline actually produces.
    pos_dir = tmp_path / "positive"
    neg_dir = tmp_path / "negative"
    pos_dir.mkdir()
    neg_dir.mkdir()
    train_pos = pos_dir / "train_pos.wav"
    val_pos = pos_dir / "val_pos.wav"
    train_neg = neg_dir / "train_neg.wav"
    val_neg = neg_dir / "val_neg.wav"
    for p in (train_pos, val_pos, train_neg, val_neg):
        sf.write(str(p), np.zeros(100, dtype=np.float32), 16000)

    manifest = {
        "train": {"positive": [str(train_pos)], "negative": [str(train_neg)]},
        "val": {"positive": [str(val_pos)], "negative": [str(val_neg)]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    report_path = tmp_path / "report.json"
    evaluate.main(
        argv=[
            "--model", "unused.tflite",
            "--manifest", str(manifest_path),
            "--split", "val",
            "--threshold", "0.5",
            "--report-out", str(report_path),
        ],
        scorer_factory=lambda model_path: _FakeScorer(model_path),
    )

    report = json.loads(report_path.read_text())
    # Only the val-split paths should have been scored -- the train-split
    # paths (also present in the manifest) must not leak in.
    assert report["num_positive"] == 1
    assert report["num_negative"] == 1
    assert report["false_reject_rate"] == 0.0
    assert report["false_accept_rate"] == 0.0


def test_main_rejects_mixing_manifest_and_directory_modes(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"val": {"positive": [], "negative": []}}))

    with pytest.raises(SystemExit):
        evaluate.main(
            argv=[
                "--model", "unused.tflite",
                "--manifest", str(manifest_path),
                "--split", "val",
                "--positive-dir", str(tmp_path),
                "--negative-dir", str(tmp_path),
                "--report-out", str(tmp_path / "report.json"),
            ],
            scorer_factory=lambda model_path: _FakeScorer(model_path),
        )


def test_main_rejects_neither_manifest_nor_directory_mode(tmp_path):
    with pytest.raises(SystemExit):
        evaluate.main(
            argv=[
                "--model", "unused.tflite",
                "--report-out", str(tmp_path / "report.json"),
            ],
            scorer_factory=lambda model_path: _FakeScorer(model_path),
        )
