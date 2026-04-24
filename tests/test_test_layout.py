from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def iter_project_test_files() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("test_*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    )


def test_all_project_tests_live_in_tests_directories():
    misplaced = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in iter_project_test_files()
        if path.parent.name != "tests"
    ]

    assert misplaced == []
