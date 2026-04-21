from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from mix.combine import sentiment_compare
from mix.combine.sentiment_compare import load_strategy_xlsx, merge_strategies, write_html_report


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


def test_build_browser_title_from_settings(tmp_path: Path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "\n".join(
            [
                "common:",
                "  ticker: 'MIX'",
                "sentiment_gemma:",
                "  sentiment_model: 'gemma3:12b'",
                "sentiment_qwen:",
                "  sentiment_model: 'qwen3:14b'",
            ]
        ),
        encoding="utf-8",
    )

    title = sentiment_compare.build_browser_title(settings_path)

    assert title == "MIX | gemma3:12b | qwen3:14b"


def test_build_page_heading_from_settings(tmp_path: Path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "\n".join(
            [
                "common:",
                "  ticker: 'MIX'",
                "sentiment_gemma:",
                "  sentiment_model: 'gemma3:12b'",
                "sentiment_qwen:",
                "  sentiment_model: 'qwen3:14b'",
            ]
        ),
        encoding="utf-8",
    )

    heading = sentiment_compare.build_page_heading(settings_path)

    assert heading == "Сводная статистика Gemma (gemma3:12b) vs Qwen (qwen3:14b) — MIX"


def test_build_test_period_from_merged_dates():
    merged = pd.DataFrame(
        [
            {"date": pd.to_datetime("2026-04-22").date()},
            {"date": pd.to_datetime("2026-04-20").date()},
            {"date": pd.to_datetime("2026-04-21").date()},
        ]
    )

    period = sentiment_compare.build_test_period(merged)

    assert period == "Период тестирования: 2026-04-20 - 2026-04-22"


def test_compare_figures_use_configured_ticker_in_titles():
    merged = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2026-04-20").date(),
                "date_dt": pd.to_datetime("2026-04-20"),
                "pnl_gemma": 10.0,
                "pnl_qwen": -4.0,
                "pnl_combined": 3.0,
                "cum_gemma": 10.0,
                "cum_qwen": -4.0,
                "cum_combined": 3.0,
            },
            {
                "date": pd.to_datetime("2026-04-21").date(),
                "date_dt": pd.to_datetime("2026-04-21"),
                "pnl_gemma": -2.0,
                "pnl_qwen": 8.0,
                "pnl_combined": 3.0,
                "cum_gemma": 8.0,
                "cum_qwen": 4.0,
                "cum_combined": 6.0,
            },
        ]
    )

    fig_compare = sentiment_compare.build_compare_figure(merged, "MIX")
    fig, fig_stats, fig_table = sentiment_compare.build_combined_report_components(merged, "MIX")

    titles = [
        fig_compare.layout.title.text,
        fig.layout.title.text,
        fig_stats.layout.title.text,
        fig_table.layout.title.text,
    ]
    assert all("MIX" in title for title in titles)
    assert all("RTS" not in title for title in titles)


def test_write_html_report_uses_titles_and_period(tmp_path: Path):
    output_html = tmp_path / "sentiment_compare.html"

    write_html_report(
        output_html,
        "<table></table>",
        go.Figure(),
        go.Figure(),
        go.Figure(),
        go.Figure(),
        browser_title="MIX | gemma3:12b | qwen3:14b",
        page_heading="Сводная статистика Gemma (gemma3:12b) vs Qwen (qwen3:14b) — MIX",
        test_period="Период тестирования: 2026-04-20 - 2026-04-22",
    )

    html = output_html.read_text(encoding="utf-8")
    assert "<title>MIX | gemma3:12b | qwen3:14b</title>" in html
    assert "Сводная статистика Gemma (gemma3:12b) vs Qwen (qwen3:14b) — MIX" in html
    assert "Период тестирования: 2026-04-20 - 2026-04-22" in html
