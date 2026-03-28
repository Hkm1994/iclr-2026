from training.hf_progress import silence_hf_download_progress


def test_silence_hf_download_progress_runs():
    silence_hf_download_progress()
