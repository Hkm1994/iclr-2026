"""Load GRaM warped-ifw style datasets: one sample per .npz at the Hub dataset repo root."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

_NPZ_UNSUPPORTED_HINT = "no (supported) data files"


def hub_dataset_has_only_npz_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return _NPZ_UNSUPPORTED_HINT in msg or "couldn't find any data file" in msg


def list_npz_filenames(
    repo_id: str,
    *,
    revision: str | None = None,
) -> list[str]:
    api = HfApi()
    names = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    return sorted(f for f in names if f.endswith(".npz"))


def sample_id_from_dataset_relpath(rel_path: str) -> str:
    p = Path(rel_path)
    return str(p.with_suffix("")).replace("\\", "/").replace("/", "-")


def row_from_npz_path(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        row = {k: np.asarray(data[k]) for k in data.files}
    row["sample_id"] = path.stem
    return row


def npz_row_from_hub_file(
    repo_id: str,
    rel_path: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    path = hf_hub_download(
        repo_id,
        rel_path,
        repo_type="dataset",
        revision=revision,
    )
    row = row_from_npz_path(path)
    row["sample_id"] = sample_id_from_dataset_relpath(rel_path)
    return row


def iter_npz_root_samples(
    repo_id: str,
    *,
    revision: str | None = None,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> Iterator[dict[str, Any]]:
    files = list_npz_filenames(repo_id, revision=revision)
    if not files:
        raise FileNotFoundError(
            f"No .npz files at dataset repo root for {repo_id!r}. "
            "Check repo_id, revision, and Hub access (token + accepted terms)."
        )
    order = list(files)
    if shuffle:
        if rng is None:
            raise ValueError("shuffle=True requires rng")
        rng.shuffle(order)
    for rel in order:
        yield npz_row_from_hub_file(repo_id, rel, revision=revision)


def load_first_npz_row(
    repo_id: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    files = list_npz_filenames(repo_id, revision=revision)
    if not files:
        raise FileNotFoundError(
            f"No .npz files in dataset repo {repo_id!r} (check access and revision)."
        )
    return npz_row_from_hub_file(repo_id, files[0], revision=revision)
