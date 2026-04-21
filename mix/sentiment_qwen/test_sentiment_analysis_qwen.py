from pathlib import Path

import pandas as pd

from mix.sentiment_qwen import sentiment_analysis_qwen as saq
from mix.sentiment_qwen.sentiment_analysis_qwen import (
    compute_content_hash,
    migrate_legacy_model_pkl,
    parse_sentiment_strict,
    resolve_model_output_pkl,
    should_process_file,
)


def test_parse_sentiment_strict_accepts_only_single_numeric_value():
    assert parse_sentiment_strict("7") == 7
    assert parse_sentiment_strict(" 4.6 \n") == 5
    assert parse_sentiment_strict("-10.2") == -10
    assert parse_sentiment_strict("11") == 10
    assert parse_sentiment_strict("Ответ: 7") is None
    assert parse_sentiment_strict("7/10") is None
    assert parse_sentiment_strict("от 3 до 5") is None


def test_should_process_file_uses_content_hash(tmp_path: Path):
    md_file = tmp_path / "2026-04-20_news.md"
    md_file.write_text("first version", encoding="utf-8")

    file_path = str(md_file.resolve())
    first_hash = compute_content_hash(md_file)
    existing_df = pd.DataFrame(
        [
            {
                "file_path": file_path,
                "content_hash": first_hash,
                "source_date": "2026-04-20",
            }
        ]
    )

    assert should_process_file(md_file, existing_df) is False

    md_file.write_text("second version", encoding="utf-8")
    assert should_process_file(md_file, existing_df) is True


def test_resolve_model_output_pkl_adds_safe_model_suffix(tmp_path: Path):
    base_path = tmp_path / "sentiment_scores.pkl"

    assert resolve_model_output_pkl(base_path, "qwen3:14b") == tmp_path / "sentiment_scores_qwen3_14b.pkl"
    assert resolve_model_output_pkl(base_path, "qwen2.5:14b") == tmp_path / "sentiment_scores_qwen2.5_14b.pkl"


def test_migrate_legacy_model_pkl_copies_only_matching_model(tmp_path: Path):
    legacy_path = tmp_path / "sentiment_scores.pkl"
    model_path = tmp_path / "sentiment_scores_qwen3_14b.pkl"
    pd.DataFrame([{"model": "qwen3:14b", "sentiment": 5}]).to_pickle(legacy_path)

    migrate_legacy_model_pkl(legacy_path, model_path, "qwen3:14b")

    assert model_path.exists()
    assert pd.read_pickle(model_path).iloc[0]["sentiment"] == 5

    model_path.unlink()
    pd.DataFrame([{"model": "qwen2.5:14b", "sentiment": -1}]).to_pickle(legacy_path)
    migrate_legacy_model_pkl(legacy_path, model_path, "qwen3:14b")

    assert not model_path.exists()


def test_main_use_cache_keeps_unchanged_latest_record(tmp_path: Path, monkeypatch):
    md_file = tmp_path / "2026-04-20_news.md"
    md_file.write_text("first version", encoding="utf-8")
    output_pkl = tmp_path / "sentiment_scores.pkl"
    content_hash = compute_content_hash(md_file)
    existing_df = pd.DataFrame(
        [
            {
                "file_path": str(md_file.resolve()),
                "content_hash": content_hash,
                "source_date": "2026-04-20",
                "ticker": "MIX",
                "model": "test-model",
                "prompt": "prompt",
                "prompt_tokens": 1,
                "raw_response": "5",
                "sentiment": 5,
                "processed_at": pd.Timestamp("2026-04-20T00:00:00Z"),
            }
        ]
    )
    existing_df.to_pickle(output_pkl)

    calls = {"run_ollama": 0}

    def fake_run_ollama(**kwargs):
        calls["run_ollama"] += 1
        return "7"

    monkeypatch.setattr(
        saq,
        "load_settings",
        lambda: {
            "ticker": "MIX",
            "md_path": str(tmp_path),
            "sentiment_output_pkl": str(output_pkl),
            "sentiment_model": "test-model",
        },
    )
    monkeypatch.setattr(saq, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(saq, "get_ollama_processor_status", lambda model: "not loaded")
    monkeypatch.setattr(saq, "run_ollama", fake_run_ollama)

    saq.main(
        output_pkl=output_pkl,
        model="test-model",
        keepalive="5m",
        token_limit=1000,
        prompt_template="{news_text}",
        use_cache=True,
    )

    assert calls["run_ollama"] == 0


def test_main_uses_cache_setting_when_cli_option_is_omitted(tmp_path: Path, monkeypatch):
    md_file = tmp_path / "2026-04-20_news.md"
    md_file.write_text("first version", encoding="utf-8")
    output_pkl = tmp_path / "sentiment_scores.pkl"
    existing_df = pd.DataFrame(
        [
            {
                "file_path": str(md_file.resolve()),
                "content_hash": compute_content_hash(md_file),
                "source_date": "2026-04-20",
                "ticker": "MIX",
                "model": "test-model",
                "prompt": "prompt",
                "prompt_tokens": 1,
                "raw_response": "5",
                "sentiment": 5,
                "processed_at": pd.Timestamp("2026-04-20T00:00:00Z"),
            }
        ]
    )
    existing_df.to_pickle(output_pkl)

    calls = {"run_ollama": 0}

    def fake_run_ollama(**kwargs):
        calls["run_ollama"] += 1
        return "7"

    monkeypatch.setattr(
        saq,
        "load_settings",
        lambda: {
            "ticker": "MIX",
            "md_path": str(tmp_path),
            "sentiment_output_pkl": str(output_pkl),
            "sentiment_model": "test-model",
            "use_cache": False,
        },
    )
    monkeypatch.setattr(saq, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(saq, "get_ollama_processor_status", lambda model: "not loaded")
    monkeypatch.setattr(saq, "run_ollama", fake_run_ollama)

    saq.main(
        output_pkl=output_pkl,
        model="test-model",
        keepalive="5m",
        token_limit=1000,
        prompt_template="{news_text}",
        use_cache=None,
    )

    assert calls["run_ollama"] == 1
