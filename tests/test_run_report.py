from __future__ import annotations

from pathlib import Path

import run_report


def test_hard_steps_include_rules_recommendation_after_group_stats():
    steps = [path.relative_to(run_report.ROOT).as_posix() for path in run_report.HARD_STEPS]

    expected_sequences = [
        [
            "rts/sentiment_gemma/sentiment_group_stats.py",
            "rts/sentiment_gemma/rules_recommendation.py",
            "rts/sentiment_gemma/sentiment_backtest.py",
        ],
        [
            "rts/sentiment_qwen/sentiment_group_stats.py",
            "rts/sentiment_qwen/rules_recommendation.py",
            "rts/sentiment_qwen/sentiment_backtest.py",
        ],
        [
            "mix/sentiment_gemma/sentiment_group_stats.py",
            "mix/sentiment_gemma/rules_recommendation.py",
            "mix/sentiment_gemma/sentiment_backtest.py",
        ],
        [
            "mix/sentiment_qwen/sentiment_group_stats.py",
            "mix/sentiment_qwen/rules_recommendation.py",
            "mix/sentiment_qwen/sentiment_backtest.py",
        ],
    ]

    for sequence in expected_sequences:
        positions = [steps.index(item) for item in sequence]
        assert positions == sorted(positions)
