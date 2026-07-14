import numpy as np
import soundfile as sf

from gen_real_positive_variants import gen_real_positive_variants


def test_generates_variants_16k_mono(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    sf.write(src / "a.wav", (0.2 * np.sin(np.linspace(0, 200, 16000))).astype("float32"), 16000)
    out = tmp_path / "out"

    n = gen_real_positive_variants(str(src), str(out), variants_per_clip=5)

    assert n == 5
    files = list(out.glob("*.wav"))
    assert len(files) == 5
    data, sr = sf.read(files[0])
    assert sr == 16000 and data.ndim == 1
