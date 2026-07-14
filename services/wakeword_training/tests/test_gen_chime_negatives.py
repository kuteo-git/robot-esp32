import numpy as np
import soundfile as sf

from gen_chime_negatives import gen_chime_negatives


def test_generates_16k_mono_variants(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    sf.write(src / "chime.wav", (0.2 * np.sin(np.linspace(0, 50, 8000))).astype("float32"), 16000)
    out = tmp_path / "out"

    n = gen_chime_negatives(str(src), str(out), variants_per_clip=5)

    assert n == 5
    files = list(out.glob("*.wav"))
    assert len(files) == 5
    data, sr = sf.read(files[0])
    assert sr == 16000 and data.ndim == 1


def test_resamples_24k_source_to_16k(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # 24kHz mono source (like end_of_request.wav) must come out at 16kHz.
    sf.write(src / "c24.wav", (0.1 * np.sin(np.linspace(0, 30, 12000))).astype("float32"), 24000)
    out = tmp_path / "out"

    gen_chime_negatives(str(src), str(out), variants_per_clip=2)

    _, sr = sf.read(next(out.glob("*.wav")))
    assert sr == 16000
