from datetime import date
from pathlib import Path

import pandas as pd

from mix.sentiment_gemma.sentiment_group_stats import (
    build_follow_trades,
    group_by_sentiment,
    index_by_date,
    resolve_group_stats_output_xlsx,
)


def test_index_by_date_uses_next_body_column():
    df = pd.DataFrame(
        [
            {"source_date": date(2026, 4, 20), "sentiment": 3, "next_body": 12.5},
            {"source_date": date(2026, 4, 21), "sentiment": -2, "next_body": -5.0},
        ]
    )

    indexed = index_by_date(df)

    assert list(indexed.columns) == ["sentiment", "next_body"]


def test_build_follow_trades_calculates_pnl_from_next_body():
    aggregated = pd.DataFrame(
        [
            {"source_date": date(2026, 4, 20), "sentiment": 3, "next_body": 10.0},
            {"source_date": date(2026, 4, 21), "sentiment": -2, "next_body": -7.0},
        ]
    ).set_index("source_date")

    trades = build_follow_trades(aggregated, quantity=2)

    assert trades["direction"].tolist() == ["LONG", "SHORT"]
    assert trades["next_body"].tolist() == [10.0, -7.0]
    assert trades["pnl"].tolist() == [20.0, 14.0]


def test_build_follow_trades_treats_zero_as_long_and_groups_it():
    aggregated = pd.DataFrame(
        [
            {"source_date": date(2026, 4, 20), "sentiment": 0, "next_body": 10.0},
        ]
    ).set_index("source_date")

    trades = build_follow_trades(aggregated, quantity=2)
    grouped = group_by_sentiment(trades)

    assert trades["direction"].tolist() == ["LONG"]
    assert trades["pnl"].tolist() == [20.0]
    zero_row = grouped.loc[grouped["sentiment"] == 0.0].iloc[0]
    assert zero_row["trades"] == 1
    assert zero_row["total_pnl"] == 20.0


def test_resolve_group_stats_output_xlsx_uses_settings_filename():
    output_dir = Path("group_stats")

    result = resolve_group_stats_output_xlsx(
        {"group_stats_output_xlsx": "sentiment_group_stats.xlsx"},
        output_dir,
    )

    assert result == output_dir / "sentiment_group_stats.xlsx"
