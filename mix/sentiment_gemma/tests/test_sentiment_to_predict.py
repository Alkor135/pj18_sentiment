import os
import importlib
from datetime import date, datetime
from pathlib import Path

import pytest

from mix.sentiment_gemma.sentiment_to_predict import (
    load_settings,
    resolve_direction,
    resolve_sentiment_pkl,
    should_rewrite_existing_predict,
)


def test_resolve_direction_treats_zero_by_rule_action():
    assert resolve_direction(0.0, "follow") == "up"
    assert resolve_direction(0.0, "invert") == "down"


def test_load_settings_uses_sentiment_gemma_section(tmp_path: Path):
    settings_yaml = tmp_path / "settings.yaml"
    settings_yaml.write_text(
        """
common:
  ticker: MIX
  ticker_lc: mix
  time_start: "21:00:00"
sentiment_gemma:
  predict_path: "C:/predict/{ticker_lc}_gemma"
  sentiment_output_pkl: "{ticker_lc}/sentiment_gemma/sentiment_scores.pkl"
""".lstrip(),
        encoding="utf-8",
    )

    settings = load_settings(settings_yaml)

    assert settings["predict_path"] == "C:/predict/mix_gemma"
    assert settings["sentiment_output_pkl"] == "mix/sentiment_gemma/sentiment_scores.pkl"


def test_resolve_sentiment_pkl_adds_model_suffix_for_relative_base_path(tmp_path: Path):
    settings = {
        "sentiment_model": "gemma3:12b",
        "sentiment_output_pkl": "sentiment_scores.pkl",
    }

    assert resolve_sentiment_pkl(settings, tmp_path) == tmp_path / "sentiment_scores_gemma3_12b.pkl"


def test_should_rewrite_existing_predict_when_file_is_before_time_start(tmp_path: Path):
    out_file = tmp_path / "2026-04-22.txt"
    out_file.write_text("old test prediction", encoding="utf-8")
    old_mtime = datetime(2026, 4, 22, 20, 59, 59).timestamp()
    os.utime(out_file, (old_mtime, old_mtime))

    assert should_rewrite_existing_predict(out_file, date(2026, 4, 22), "21:00:00") is True


def test_should_keep_existing_predict_when_file_is_after_time_start(tmp_path: Path):
    out_file = tmp_path / "2026-04-22.txt"
    out_file.write_text("fresh prediction", encoding="utf-8")
    fresh_mtime = datetime(2026, 4, 22, 21, 0, 1).timestamp()
    os.utime(out_file, (fresh_mtime, fresh_mtime))

    assert should_rewrite_existing_predict(out_file, date(2026, 4, 22), "21:00:00") is False


@pytest.mark.parametrize(
    "module_name",
    [
        "rts.sentiment_gemma.sentiment_to_predict",
        "rts.sentiment_qwen.sentiment_to_predict",
        "mix.sentiment_gemma.sentiment_to_predict",
        "mix.sentiment_qwen.sentiment_to_predict",
    ],
)
def test_main_reads_rules_yaml_from_own_model_directory(module_name: str, tmp_path: Path, monkeypatch):
    module = importlib.import_module(module_name)
    expected_rules_path = Path(module.__file__).resolve().parent / "rules.yaml"
    captured = {}

    monkeypatch.setattr(module, "setup_logging", lambda: type("Logger", (), {
        "info": lambda self, *args, **kwargs: None,
        "error": lambda self, *args, **kwargs: None,
        "exception": lambda self, *args, **kwargs: None,
    })())
    monkeypatch.setattr(module, "load_settings", lambda path: {
        "predict_path": str(tmp_path / "predict"),
        "time_start": "21:00:00",
        "sentiment_model": "test:model",
        "sentiment_output_pkl": "sentiment_scores.pkl",
    })
    monkeypatch.setattr(module, "write_predict", lambda *args, **kwargs: None)

    def capture_rules_path(path: Path):
        captured["path"] = path
        raise RuntimeError("stop after rules path capture")

    monkeypatch.setattr(module, "load_rules", capture_rules_path)

    assert module.main() == 0
    assert captured["path"] == expected_rules_path
