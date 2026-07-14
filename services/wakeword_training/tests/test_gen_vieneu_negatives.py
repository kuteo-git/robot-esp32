import io
from unittest import mock

import numpy as np
import soundfile as sf

from gen_vieneu_negatives import gen_vieneu_negatives


def _fake_wav_bytes(n=16000, sr=16000, channels=1):
    buf = io.BytesIO()
    shape = (n,) if channels == 1 else (n, channels)
    sf.write(buf, np.zeros(shape, "float32"), sr, format="WAV")
    return buf.getvalue()


def test_writes_one_wav_per_sentence_voice(tmp_path):
    with mock.patch("gen_vieneu_negatives._synth", return_value=_fake_wav_bytes()):
        n = gen_vieneu_negatives(["câu một", "câu hai"], ["Ngọc Lan"], str(tmp_path))
    assert n == 2
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_resamples_and_downmixes_to_16k_mono(tmp_path):
    with mock.patch("gen_vieneu_negatives._synth", return_value=_fake_wav_bytes(24000, sr=24000, channels=2)):
        gen_vieneu_negatives(["x"], ["V"], str(tmp_path))
    data, sr = sf.read(next(tmp_path.glob("*.wav")))
    assert sr == 16000 and data.ndim == 1
