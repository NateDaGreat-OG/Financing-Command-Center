"""Regression tests for importing the project package from the repository root."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_project_app_imports_from_repo_root():
    """`import project.app` should work when only the repository root is on sys.path."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import project.app; print(project.app.app.name)"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "project.app"
