from pathlib import Path

import pandas as pd

from mix.combine.sentiment_compare import load_strategy_xlsx, merge_strategies


def test_load_strategy_xlsx_reads_source_date_and_pnl(tmp_path: Path):
    xlsx_path = tmp_path / "gemma.xlsx"
    pd.DataFrame(
        [
            {"source_date": "2026-04-20", "pnl": 10.0},
            {"source_date": "2026-04-21", "pnl": -5.0},
        ]
    ).to_excel(xlsx_path, index=False)

    result = load_strategy_xlsx(xlsx_path, "pnl_gemma")

    assert list(result.columns) == ["date", "pnl_gemma"]
    assert result["pnl_gemma"].tolist() == [10.0, -5.0]


def test_merge_strategies_combines_gemma_and_qwen_by_date():
    gemma_df = pd.DataFrame(
        [
            {"date": pd.to_datetime("2026-04-20").date(), "pnl_gemma": 10.0},
            {"date": pd.to_datetime("2026-04-21").date(), "pnl_gemma": -4.0},
        ]
    )
    qwen_df = pd.DataFrame(
        [
            {"date": pd.to_datetime("2026-04-21").date(), "pnl_qwen": 8.0},
            {"date": pd.to_datetime("2026-04-22").date(), "pnl_qwen": 6.0},
        ]
    )

    merged = merge_strategies(gemma_df, qwen_df)

    assert merged["pnl_gemma"].tolist() == [10.0, -4.0, 0.0]
    assert merged["pnl_qwen"].tolist() == [0.0, 8.0, 6.0]
    assert merged["pnl_combined"].tolist() == [5.0, 2.0, 3.0]
