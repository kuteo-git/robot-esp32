import json

from prepare_manifest import build_manifest, write_manifest


def _make_wavs(dir_path, n):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (dir_path / f"clip_{i}.wav").write_bytes(b"")
    return dir_path


def test_build_manifest_splits_without_overlap_and_covers_all_files(tmp_path):
    pos_dir = _make_wavs(tmp_path / "positive", 20)
    neg_dir_a = _make_wavs(tmp_path / "neg_a", 10)
    neg_dir_b = _make_wavs(tmp_path / "neg_b", 10)

    manifest = build_manifest(pos_dir, [neg_dir_a, neg_dir_b], val_fraction=0.2, seed=0)

    train_pos = set(manifest["train"]["positive"])
    val_pos = set(manifest["val"]["positive"])
    train_neg = set(manifest["train"]["negative"])
    val_neg = set(manifest["val"]["negative"])

    assert len(train_pos) + len(val_pos) == 20
    assert train_pos.isdisjoint(val_pos)
    assert len(train_neg) + len(val_neg) == 20
    assert train_neg.isdisjoint(val_neg)
    assert len(val_pos) == 4  # 20 * 0.2
    assert len(val_neg) == 4  # 20 * 0.2


def test_build_manifest_is_deterministic_given_seed(tmp_path):
    pos_dir = _make_wavs(tmp_path / "positive", 20)
    neg_dir = _make_wavs(tmp_path / "neg", 20)

    m1 = build_manifest(pos_dir, [neg_dir], val_fraction=0.25, seed=42)
    m2 = build_manifest(pos_dir, [neg_dir], val_fraction=0.25, seed=42)
    assert m1 == m2


def test_write_manifest_writes_valid_json(tmp_path):
    pos_dir = _make_wavs(tmp_path / "positive", 4)
    neg_dir = _make_wavs(tmp_path / "neg", 4)
    manifest = build_manifest(pos_dir, [neg_dir], val_fraction=0.25, seed=0)

    out_path = tmp_path / "manifest.json"
    write_manifest(manifest, out_path)

    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == manifest


def test_build_manifest_does_not_pick_up_mmap_feature_folders(tmp_path):
    """negative_standard-style Ragged Mmap folders (data/, starts/, ends/, shapes/,
    flattened_shapes/, shapes_are_flat.ninja -- no .wav files) must contribute zero
    entries if ever passed in by mistake, since build_manifest() only globs raw
    ``.wav`` files. This guards the exact bug Task 7 identified: silently treating
    pre-extracted-feature folders as if they were raw audio directories.
    """
    pos_dir = _make_wavs(tmp_path / "positive", 4)
    mmap_like_dir = tmp_path / "dinner_party_mmap"
    mmap_like_dir.mkdir(parents=True, exist_ok=True)
    (mmap_like_dir / "data").mkdir()
    (mmap_like_dir / "starts").mkdir()
    (mmap_like_dir / "shapes_are_flat.ninja").write_bytes(b"")

    manifest = build_manifest(pos_dir, [mmap_like_dir], val_fraction=0.25, seed=0)

    assert manifest["train"]["negative"] == []
    assert manifest["val"]["negative"] == []
