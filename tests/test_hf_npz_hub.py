import numpy as np

from training.hf_npz_hub import (
    hub_dataset_has_only_npz_error,
    row_from_npz_path,
    sample_id_from_dataset_relpath,
)


def test_hub_dataset_has_only_npz_error_detects_message():
    assert hub_dataset_has_only_npz_error(
        RuntimeError("No (supported) data files found in gram-competition/warped-ifw")
    )
    assert not hub_dataset_has_only_npz_error(RuntimeError("401 Unauthorized"))


def test_sample_id_from_dataset_relpath_nested():
    assert sample_id_from_dataset_relpath("1021_1-0.npz") == "1021_1-0"
    assert sample_id_from_dataset_relpath("a/b-1.npz") == "a-b-1"


def test_row_from_npz_path_roundtrip(tmp_path):
    p = tmp_path / "sim-0.npz"
    np.savez(
        p,
        t=np.zeros(10, dtype=np.float32),
        pos=np.zeros((100, 3), dtype=np.float32),
        idcs_airfoil=np.array([0, 1], dtype=np.int64),
        velocity_in=np.zeros((5, 100, 3), dtype=np.float32),
        velocity_out=np.zeros((5, 100, 3), dtype=np.float32),
    )
    row = row_from_npz_path(p)
    assert row["sample_id"] == "sim-0"
    assert row["t"].shape == (10,)
    assert row["pos"].shape == (100, 3)
