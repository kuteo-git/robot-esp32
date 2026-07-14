import io
from unittest import mock
import numpy as np, soundfile as sf
from gen_vieneu_positives import gen_vieneu_positives

def _fake():
    buf=io.BytesIO(); sf.write(buf, np.zeros(16000,"float32"),16000,format="WAV"); return buf.getvalue()

def test_generates_positives_16k_mono(tmp_path):
    with mock.patch("gen_vieneu_positives._synth", return_value=_fake()):
        n = gen_vieneu_positives(str(tmp_path), variants_per_clip=2)
    assert n == 10*5*2  # voices*phrases*variants
    d,sr=sf.read(next(tmp_path.glob("*.wav"))); assert sr==16000 and d.ndim==1
