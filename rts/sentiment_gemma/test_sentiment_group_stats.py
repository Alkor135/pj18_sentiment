from datetime import date

import pandas as pd

from rts.sentiment_gemma.sentiment_group_stats import build_follow_trades, index_by_date


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
