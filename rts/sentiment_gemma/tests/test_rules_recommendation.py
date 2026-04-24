from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "rules_recommendation.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("rules_recommendation_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_rules_recommendation_matches_reference_example():
    module = load_module()
    grouped = pd.DataFrame(
        {
            "sentiment": list(range(-10, 11)),
            "total_pnl": [
                0.0,
                0.0,
                2.09,
                4.25,
                5.76,
                -2.32,
                -1.59,
                -19.92,
                4.08,
                0.0,
                0.0,
                0.0,
                5.53,
                3.55,
                -2.39,
                -1.60,
                1.85,
                -1.47,
                -4.52,
                0.0,
                0.0,
            ],
        }
    )

    rules = module.build_rules_recommendation(grouped)

    expected_actions = {
        -10: "follow",
        -9: "follow",
        -8: "follow",
        -7: "follow",
        -6: "follow",
        -5: "invert",
        -4: "invert",
        -3: "invert",
        -2: "follow",
        -1: "follow",
        0: "follow",
        1: "follow",
        2: "follow",
        3: "follow",
        4: "invert",
        5: "invert",
        6: "follow",
        7: "invert",
        8: "invert",
        9: "invert",
        10: "invert",
    }

    assert rules == [
        {"min": sentiment, "max": sentiment, "action": expected_actions[sentiment]}
        for sentiment in range(-10, 11)
    ]


def test_equal_distance_zero_uses_larger_absolute_total_pnl():
    module = load_module()
    grouped = pd.DataFrame(
        {
            "sentiment": [-2, -1, 0, 1, 2],
            "total_pnl": [0.0, -2.0, 0.0, 5.0, 0.0],
        }
    )

    assert module.recommend_action(grouped.set_index("sentiment")["total_pnl"], 0) == "follow"


def test_equal_distance_and_equal_absolute_total_pnl_checks_next_neighbors():
    module = load_module()
    grouped = pd.DataFrame(
        {
            "sentiment": [-3, -2, -1, 0, 1, 2, 3],
            "total_pnl": [0.0, -7.0, -5.0, 0.0, 5.0, 9.0, 0.0],
        }
    )

    assert module.recommend_action(grouped.set_index("sentiment")["total_pnl"], 0) == "follow"
