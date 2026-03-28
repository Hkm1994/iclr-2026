"""Optional quiet mode for Hugging Face downloads (cleaner training logs)."""

from __future__ import annotations

import os


def silence_hf_download_progress() -> None:
    """
    Turn off tqdm-style progress from huggingface_hub and datasets during this process.

    Skipped if env ``GRAM_SHOW_HF_PROGRESS=1`` (show bars anyway).
    Respects existing ``HF_*`` env if already set.
    """
    if os.environ.get("GRAM_SHOW_HF_PROGRESS") == "1":
        return
    if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") is None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if os.environ.get("HF_DATASETS_DISABLE_PROGRESS_BARS") is None:
        os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        from datasets.utils.logging import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass
