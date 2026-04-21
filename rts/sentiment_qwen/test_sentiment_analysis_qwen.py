from pathlib import Path

import pandas as pd

from rts.sentiment_qwen.sentiment_analysis_qwen import (
    compute_content_hash,
    parse_sentiment_strict,
    parse_ollama_processor_status,
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


def test_parse_ollama_processor_status_extracts_cpu_gpu_split():
    output = """NAME         ID              SIZE     PROCESSOR          CONTEXT    UNTIL
qwen3:14b    bdbd181c33f2    10 GB    11%/89% CPU/GPU    4096       4 minutes from now
"""

    assert parse_ollama_processor_status(output, "qwen3:14b") == "11%/89% CPU/GPU"
    assert parse_ollama_processor_status(output, "qwen2.5:14b") == "not loaded"
