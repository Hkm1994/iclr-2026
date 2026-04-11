import textwrap
from pathlib import Path

import numpy as np
import pytest

from training.hf_dataset import build_stream
from training.hf_npz_hub import (
    iter_npz_local_samples,
    list_npz_relpaths_local,
    load_first_npz_row_local,
    sample_id_from_dataset_relpath,
)


def _minimal_npz(path: Path) -> None:
    np.savez(
        path,
        t=np.zeros(10, dtype=np.float32),
        pos=np.zeros((100, 3), dtype=np.float32),
        idcs_airfoil=np.array([0, 1], dtype=np.int64),
        velocity_in=np.zeros((5, 100, 3), dtype=np.float32),
        velocity_out=np.zeros((5, 100, 3), dtype=np.float32),
    )


def test_list_npz_relpaths_local_sorted_nested(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    _minimal_npz(tmp_path / "z.npz")
    _minimal_npz(sub / "a.npz")
    rels = list_npz_relpaths_local(tmp_path)
    assert rels == ["sub/a.npz", "z.npz"]


def test_iter_npz_local_samples_sample_ids_match_hub_rule(tmp_path: Path) -> None:
    case = tmp_path / "shard" / "case"
    case.mkdir(parents=True)
    _minimal_npz(case / "x.npz")
    _minimal_npz(tmp_path / "rooty.npz")
    rows = list(iter_npz_local_samples(tmp_path))
    by_id = {r["sample_id"]: r for r in rows}
    assert set(by_id) == {
        sample_id_from_dataset_relpath("shard/case/x.npz"),
        sample_id_from_dataset_relpath("rooty.npz"),
    }
    for rel, row in (
        ("shard/case/x.npz", by_id[sample_id_from_dataset_relpath("shard/case/x.npz")]),
        ("rooty.npz", by_id[sample_id_from_dataset_relpath("rooty.npz")]),
    ):
        assert row["sample_id"] == sample_id_from_dataset_relpath(rel)


def test_load_first_npz_row_local_first_sorted_relpath(tmp_path: Path) -> None:
    _minimal_npz(tmp_path / "b.npz")
    _minimal_npz(tmp_path / "a.npz")
    row = load_first_npz_row_local(tmp_path)
    assert row["sample_id"] == sample_id_from_dataset_relpath("a.npz")


def test_build_stream_local_explicit_lists(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _minimal_npz(data_root / "one.npz")
    (data_root / "nested").mkdir()
    _minimal_npz(data_root / "nested" / "two.npz")
    train_ids = tmp_path / "train_ids.txt"
    id_one = sample_id_from_dataset_relpath("one.npz")
    id_two = sample_id_from_dataset_relpath("nested/two.npz")
    train_ids.write_text(f"{id_one}\n{id_two}\n", encoding="utf-8")
    cfg_path = tmp_path / "data_split.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            version: "test-local"
            dataset:
              repo_id: dummy/dummy
              revision: null
              layout: auto
              source: local
              local_path: {data_root.as_posix()}
            id_key: sample_id
            seed: 42
            split:
              mode: explicit_lists
              hf_native:
                train_split: train
                val_split: validation
                test_split: null
              hash_ids:
                hf_split: train
                val_fraction: 0.1
                test_fraction: 0.05
              explicit_lists:
                hf_split: train
                train_ids_path: {train_ids.as_posix()}
                val_ids_path: null
                test_ids_path: null
            streaming:
              shuffle_buffer_size: 100
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    out = list(build_stream(cfg_path, "train"))
    assert len(out) == 2
    assert {r["sample_id"] for r in out} == {id_one, id_two}


def test_build_stream_local_hf_native_errors(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            version: "test"
            dataset:
              repo_id: dummy/dummy
              revision: null
              layout: auto
              source: local
              local_path: /tmp/nowhere
            id_key: sample_id
            seed: 42
            split:
              mode: hf_native
              hf_native:
                train_split: train
                val_split: validation
                test_split: null
              hash_ids:
                hf_split: train
                val_fraction: 0.1
                test_fraction: 0.05
              explicit_lists:
                hf_split: train
                train_ids_path: null
                val_ids_path: null
                test_ids_path: null
            streaming:
              shuffle_buffer_size: 100
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hf_native"):
        next(build_stream(cfg_path, "train"))


def test_build_stream_local_hub_table_errors(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            version: "test"
            dataset:
              repo_id: dummy/dummy
              revision: null
              layout: hub_table
              source: local
              local_path: /tmp/nowhere
            id_key: sample_id
            seed: 42
            split:
              mode: hash_ids
              hf_native:
                train_split: train
                val_split: validation
                test_split: null
              hash_ids:
                hf_split: train
                val_fraction: 0.1
                test_fraction: 0.05
              explicit_lists:
                hf_split: train
                train_ids_path: null
                val_ids_path: null
                test_ids_path: null
            streaming:
              shuffle_buffer_size: 100
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hub_table"):
        next(build_stream(cfg_path, "train"))
