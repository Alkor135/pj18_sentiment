from pathlib import Path

import pandas as pd

from mix.sentiment_qwen.sentiment_analysis_qwen import (
    compute_content_hash,
    parse_sentiment_strict,
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
