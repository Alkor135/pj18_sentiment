from datetime import date

import pandas as pd

from rts.sentiment_qwen.sentiment_backtest import build_backtest, index_by_date


def test_index_by_date_uses_next_body():
    df = pd.DataFrame(
        [
            {"source_date": date(2026, 4, 20), "sentiment": 3, "next_body": 10.0},
            {"source_date": date(2026, 4, 21), "sentiment": -2, "next_body": -4.0},
        ]
    )

    indexed = index_by_date(df)

    assert list(indexed.columns) == ["sentiment", "next_body"]


def test_build_backtest_uses_next_body_for_follow_and_invert():
    aggregated = pd.DataFrame(
        [
            {"source_date": date(2026, 4, 20), "sentiment": 3, "next_body": 5.0},
            {"source_date": date(2026, 4, 21), "sentiment": -4, "next_body": -7.0},
        ]
    ).set_index("source_date")
    rules = [
        {"min": 3, "max": 3, "action": "follow"},
        {"min": -4, "max": -4, "action": "invert"},
    ]

    result = build_backtest(aggregated, quantity=2, rules=rules)

    assert result["direction"].tolist() == ["LONG", "LONG"]
    assert result["next_body"].tolist() == [5.0, -7.0]
    assert result["pnl"].tolist() == [10.0, -14.0]


def test_build_backtest_treats_zero_as_long_for_follow_and_short_for_invert():
    aggregated = pd.DataFrame(
        [
            {"source_date": date(2026, 4, 20), "sentiment": 0, "next_body": 5.0},
            {"source_date": date(2026, 4, 21), "sentiment": 0, "next_body": 7.0},
        ]
    ).set_index("source_date")
    rules = [
        {"min": 0, "max": 0, "action": "follow"},
        {"min": 0, "max": 0, "action": "invert"},
    ]

    follow_result = build_backtest(aggregated.iloc[[0]], quantity=2, rules=rules[:1])
    invert_result = build_backtest(aggregated.iloc[[1]], quantity=2, rules=rules[1:])

    assert follow_result["direction"].tolist() == ["LONG"]
    assert follow_result["pnl"].tolist() == [10.0]
    assert invert_result["direction"].tolist() == ["SHORT"]
    assert invert_result["pnl"].tolist() == [-14.0]
