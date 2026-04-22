from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import prepare


def write_settings(path: Path, ticker_lc: str, base_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
common:
  ticker: '{ticker_lc.upper()}'
  ticker_lc: '{ticker_lc}'
sentiment_gemma:
  predict_path: '{base_dir}/pj18/{{ticker_lc}}_sentiment_gemma'
sentiment_qwen:
  predict_path: '{base_dir}/pj18/{{ticker_lc}}_sentiment_qwen'
combined:
  predict_path: '{base_dir}/pj18/{{ticker_lc}}_sentiment_combined'
""".lstrip(),
        encoding="utf-8",
    )


def test_collect_predict_files_reads_paths_from_ticker_settings(tmp_path):
    base_dir = tmp_path / "predict_ai"
    write_settings(tmp_path / "rts" / "settings.yaml", "rts", base_dir)
    write_settings(tmp_path / "mix" / "settings.yaml", "mix", base_dir)

    files = prepare.collect_predict_files_for_date(
        tmp_path,
        date(2026, 4, 22),
    )

    assert files == [
        base_dir / "pj18" / "rts_sentiment_gemma" / "2026-04-22.txt",
        base_dir / "pj18" / "rts_sentiment_qwen" / "2026-04-22.txt",
        base_dir / "pj18" / "rts_sentiment_combined" / "2026-04-22.txt",
        base_dir / "pj18" / "mix_sentiment_gemma" / "2026-04-22.txt",
        base_dir / "pj18" / "mix_sentiment_qwen" / "2026-04-22.txt",
        base_dir / "pj18" / "mix_sentiment_combined" / "2026-04-22.txt",
    ]


def test_main_cleans_old_done_markers_after_cutoff(tmp_path, monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 4, 22, 21, 0, 5)

    log_dir = tmp_path / "log"
    state_dir = tmp_path / "trade" / "state"
    log_dir.mkdir()
    state_dir.mkdir(parents=True)
    old_marker = state_dir / "rts_EBS_sentiment_2026-04-01.done"
    old_marker.touch()

    monkeypatch.setattr(prepare, "datetime", FixedDateTime)
    monkeypatch.setattr(prepare, "LOG_DIR", log_dir)
    monkeypatch.setattr(prepare, "STATE_DIR", state_dir)

    assert prepare.main() == 0
    assert not old_marker.exists()
