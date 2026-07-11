import json

import numpy as np
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
