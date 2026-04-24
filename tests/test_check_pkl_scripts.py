from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_RELATIVE_PATHS = [
    Path("rts/sentiment_gemma/check_pkl.py"),
    Path("rts/sentiment_qwen/check_pkl.py"),
    Path("mix/sentiment_gemma/check_pkl.py"),
    Path("mix/sentiment_qwen/check_pkl.py"),
]


def load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(
        f"check_pkl_{script_path.parts[0]}_{script_path.parts[1]}",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative_path", SCRIPT_RELATIVE_PATHS)
def test_check_pkl_scripts_default_to_their_own_directory(relative_path):
    script_path = REPO_ROOT / relative_path
    assert script_path.exists()

    module = load_module(script_path)

    assert module.resolve_directory([]) == script_path.parent

    tmp_dir = script_path.parent / "__tmp_check_pkl_test__"
    tmp_dir.mkdir(exist_ok=True)
    sample_pkl = tmp_dir / "sample.pkl"
    try:
        with sample_pkl.open("wb") as file:
            pickle.dump(pd.DataFrame({"value": range(3)}), file)

        assert module.main([str(tmp_dir)]) == 0
    finally:
        sample_pkl.unlink(missing_ok=True)
        tmp_dir.rmdir()
