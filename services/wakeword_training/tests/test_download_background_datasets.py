import pytest

from download_background_datasets import (
    NEGATIVE_DATASET_FILENAMES,
    main,
    validate_dataset_dir,
)


def _make_ragged_mmap_dir(path):
    """Build a directory matching the real Ragged Mmap layout written by
    mmap_ninja.ragged.RaggedMmap (the format vendor/microWakeWord's
    kahrendt/microwakeword negative datasets are actually distributed in):
    a directory whose name ends in "_mmap", containing "data", "starts",
    "ends", "shapes", "flattened_shapes" subfolders and a
    "shapes_are_flat.ninja" marker file. See
    .venv-train/lib/python3.12/site-packages/mmap_ninja/ragged.py and
    vendor/microWakeWord/microwakeword/data.py (MmapFeatureGenerator, which
    globs "**/*_mmap/" under each split directory).
    """
    path.mkdir(parents=True)
    (path / "shapes_are_flat.ninja").write_text("1")
    for sub in ("data", "starts", "ends", "shapes", "flattened_shapes"):
        (path / sub).mkdir()
        (path / sub / "type.ninja").write_text("numpy")


def test_validate_dataset_dir_counts_ragged_mmap_folders(tmp_path):
    _make_ragged_mmap_dir(tmp_path / "speech" / "training" / "wakeword_mmap")
    _make_ragged_mmap_dir(tmp_path / "speech" / "validation" / "wakeword_mmap")
    _make_ragged_mmap_dir(tmp_path / "no_speech" / "training" / "wakeword_mmap")
    (tmp_path / "README.md").write_text("not a feature folder")

    count = validate_dataset_dir(tmp_path)
    assert count == 3


def test_validate_dataset_dir_raises_on_empty(tmp_path):
    with pytest.raises(ValueError):
        validate_dataset_dir(tmp_path)


def test_validate_dataset_dir_ignores_non_mmap_dirs(tmp_path):
    # A "training" split directory with a subfolder that does NOT end in
    # "_mmap" must not be counted as a feature folder.
    (tmp_path / "training" / "not_a_feature_dir").mkdir(parents=True)
    (tmp_path / "training" / "not_a_feature_dir" / "data.ninja").write_text("x")

    with pytest.raises(ValueError):
        validate_dataset_dir(tmp_path)


def test_main_downloads_and_extracts_each_dataset_zip(tmp_path):
    calls = []

    def fake_download_and_extract(filename, out_dir):
        calls.append((filename, out_dir))
        dataset_name = filename.replace(".zip", "")
        _make_ragged_mmap_dir(out_dir / dataset_name / "training" / "feat_mmap")

    out_dir = tmp_path / "negative_standard"
    main(argv=["--out-dir", str(out_dir)], download_and_extract=fake_download_and_extract)

    # Every dataset zip named in NEGATIVE_DATASET_FILENAMES (the four zips
    # the upstream notebook downloads: dinner_party.zip, dinner_party_eval.zip,
    # no_speech.zip, speech.zip) must be downloaded into the same --out-dir.
    assert [c[0] for c in calls] == NEGATIVE_DATASET_FILENAMES
    assert all(c[1] == out_dir for c in calls)

    # One Ragged Mmap feature folder was "extracted" per dataset zip above.
    assert validate_dataset_dir(out_dir) == len(NEGATIVE_DATASET_FILENAMES)


def test_main_defaults_out_dir_to_data_negative_standard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_download_and_extract(filename, out_dir):
        calls.append(out_dir)
        _make_ragged_mmap_dir(out_dir / filename.replace(".zip", "") / "training" / "feat_mmap")

    main(argv=[], download_and_extract=fake_download_and_extract)

    expected_out_dir = tmp_path / "data" / "negative_standard"
    assert all(c.resolve() == expected_out_dir.resolve() for c in calls)
    assert expected_out_dir.exists()
