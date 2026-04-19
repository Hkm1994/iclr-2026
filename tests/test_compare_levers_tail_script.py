"""compare_levers_tail_mlflow.py exits 0 when baseline exists (tail may be incomplete)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_compare_levers_tail_script_runs():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "compare_levers_tail_mlflow.py"
    env = {**os.environ, "PYTHONPATH": str(root)}
    r = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
