from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "check_pkl.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_pkl_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_pkl_processes_all_pickle_files(tmp_path, capsys):
    first = tmp_path / "a.pkl"
    second = tmp_path / "b.pkl"
    with first.open("wb") as file:
        pickle.dump(pd.DataFrame({"value": range(7)}), file)
    with second.open("wb") as file:
        pickle.dump(pd.DataFrame({"value": range(10, 17)}), file)

    module = load_module()

    assert module.main([str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "a.pkl" in output
    assert "b.pkl" in output
    assert "HEAD 5" in output
    assert "TAIL 5" in output
