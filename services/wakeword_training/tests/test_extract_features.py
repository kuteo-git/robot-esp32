"""Tests for extract_features.py.

These run the REAL vendored feature-extraction pipeline (Augmentation,
SpectrogramGeneration, RaggedMmap.from_generator, pymicro-features' C
microfrontend) over tiny synthetic WAV fixtures -- not mocks -- so a passing
suite here is a genuine (if small-scale) smoke test of the WAV -> Ragged
Mmap step, matching Task 9's brief: "feed a handful of tiny dummy WAV files
through the real feature-extraction function and confirm it produces a real
mmap folder."
"""
import numpy as np
from mmap_ninja.ragged import RaggedMmap
from scipy.io import wavfile

from extract_features import ManifestClips, build_augmenter, extract_all, extract_source


def _make_wav(path, duration_s=1.0, freq=440.0, seed=0):
    sr = 16000
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq * t)
    noise = 0.01 * rng.standard_normal(t.shape)
    audio = ((tone + noise) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, audio)
    return str(path)


def test_manifest_clips_yields_decoded_audio_for_explicit_paths(tmp_path):
    paths = [_make_wav(tmp_path / f"clip_{i}.wav", duration_s=0.5, freq=300 + i * 50) for i in range(3)]

    clips = ManifestClips(paths)
    assert len(clips) == 3

    yielded = list(clips.audio_generator(split="training", repeat=1))
    assert len(yielded) == 3
    for clip in yielded:
        assert isinstance(clip, np.ndarray)
        assert clip.shape[0] == 8000  # 0.5s @ 16kHz

    # repeat=2 must yield each clip twice
    assert len(list(clips.audio_generator(repeat=2))) == 6


def test_manifest_clips_get_random_clip_returns_decoded_audio(tmp_path):
    paths = [_make_wav(tmp_path / f"clip_{i}.wav", duration_s=0.3) for i in range(2)]
    clips = ManifestClips(paths)
    clip = clips.get_random_clip()
    assert isinstance(clip, np.ndarray)
    assert clip.shape[0] == 4800  # 0.3s @ 16kHz


def test_extract_source_writes_real_readable_mmap_folders(tmp_path):
    positive_paths = [
        _make_wav(tmp_path / "raw" / "positive" / f"pos_{i}.wav", duration_s=1.0, freq=300 + i * 40, seed=i)
        for i in range(3)
    ]
    manifest = {
        "train": {"positive": positive_paths[:2], "negative": []},
        "val": {"positive": positive_paths[2:], "negative": []},
    }

    out_dir = tmp_path / "features" / "positive"
    augmenter = build_augmenter()  # no RIR/background dirs -> those augmentations no-op
    counts = extract_source(manifest, "positive", out_dir, "positive", augmenter)

    # "training" split repeats each of the 2 train clips twice (SPLIT_PLAN),
    # so it must have written more spectrograms than there were source clips.
    assert counts["training"] > 0
    assert counts["validation"] > 0
    assert counts["testing"] > 0

    for split in ("training", "validation", "testing"):
        mmap_dir = out_dir / split / "positive_mmap"
        assert mmap_dir.is_dir()
        loaded = RaggedMmap(str(mmap_dir))
        assert len(loaded) == counts[split]
        # Each spectrogram is a real 2D (time, 40 mel channels) float/uint16 array.
        assert loaded[0].ndim == 2
        assert loaded[0].shape[1] == 40


def test_extract_source_skips_split_with_no_source_clips(tmp_path):
    positive_paths = [_make_wav(tmp_path / "raw" / f"pos_{i}.wav", duration_s=0.5) for i in range(2)]
    manifest = {
        "train": {"positive": positive_paths, "negative": []},
        "val": {"positive": [], "negative": []},  # no val clips at all
    }

    out_dir = tmp_path / "features" / "positive"
    augmenter = build_augmenter()
    counts = extract_source(manifest, "positive", out_dir, "positive", augmenter)

    assert counts["validation"] == 0
    assert counts["testing"] == 0
    # No mmap folder should be written for splits with zero source clips.
    assert not (out_dir / "validation").exists()
    assert not (out_dir / "testing").exists()
    assert (out_dir / "training" / "positive_mmap").is_dir()


def test_extract_all_produces_separate_positive_and_negative_vi_folders(tmp_path):
    pos_paths = [_make_wav(tmp_path / "raw" / f"pos_{i}.wav", duration_s=0.4, freq=250) for i in range(2)]
    neg_paths = [_make_wav(tmp_path / "raw" / f"neg_{i}.wav", duration_s=0.4, freq=900) for i in range(2)]
    manifest = {
        "train": {"positive": pos_paths, "negative": neg_paths},
        "val": {"positive": [], "negative": []},
    }

    out_dir = tmp_path / "features"
    results = extract_all(manifest, out_dir)

    assert results["positive"]["training"] > 0
    assert results["negative_vi"]["training"] > 0
    assert (out_dir / "positive" / "training" / "positive_mmap").is_dir()
    assert (out_dir / "negative_vi" / "training" / "negative_vi_mmap").is_dir()
