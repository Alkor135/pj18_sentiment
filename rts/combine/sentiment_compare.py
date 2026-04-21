"""Сравнивает результаты backtest двух sentiment-стратегий: Gemma и Qwen.

Скрипт читает два XLSX-файла с результатами backtest:
- `rts/sentiment_gemma/backtest/sentiment_backtest_results.xlsx`
- `rts/sentiment_qwen/backtest/sentiment_backtest_results.xlsx`

Далее он:
- объединяет стратегии по `source_date`;
- строит отдельные equity-кривые Gemma, Qwen и их простой комбинации;
- считает подробную статистику для комбинированной стратегии;
- сохраняет HTML-отчёт в `rts/combine/plots`.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import typer
import yaml

TICKER_DIR = Path(__file__).resolve().parents[1]

GEMMA_XLSX = TICKER_DIR / "sentiment_gemma" / "backtest" / "sentiment_backtest_results.xlsx"
QWEN_XLSX = TICKER_DIR / "sentiment_qwen" / "backtest" / "sentiment_backtest_results.xlsx"
OUTPUT_HTML = Path(__file__).resolve().parent / "plots" / "sentiment_compare.html"
SETTINGS_YAML = TICKER_DIR / "settings.yaml"

app = typer.Typer(help="Сравнение sentiment_gemma и sentiment_qwen по результатам backtest.")


def load_report_labels(settings_path: Path = SETTINGS_YAML) -> dict[str, str]:
    """Читает ticker и модели sentiment-стратегий для подписей отчёта."""
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    common = raw.get("common") or {}
    gemma = raw.get("sentiment_gemma") or {}
    qwen = raw.get("sentiment_qwen") or {}
    return {
        "ticker": str(common.get("ticker", TICKER_DIR.name.upper())),
        "gemma_model": str(gemma.get("sentiment_model", "gemma")),
        "qwen_model": str(qwen.get("sentiment_model", "qwen")),
    }


def build_browser_title(settings_path: Path = SETTINGS_YAML) -> str:
    """Формирует title вкладки браузера из ticker и моделей sentiment-стратегий."""
    labels = load_report_labels(settings_path)
    ticker = labels["ticker"]
    gemma_model = labels["gemma_model"]
    qwen_model = labels["qwen_model"]
    return f"{ticker} | {gemma_model} | {qwen_model}"


def build_page_heading(settings_path: Path = SETTINGS_YAML) -> str:
    """Формирует видимый заголовок страницы с моделями рядом с названиями стратегий."""
    labels = load_report_labels(settings_path)
    return (
        f"Сводная статистика Gemma ({labels['gemma_model']}) "
        f"vs Qwen ({labels['qwen_model']}) — {labels['ticker']}"
    )


def build_test_period(merged: pd.DataFrame) -> str:
    """Формирует период тестирования по фактическим датам сравнения."""
    dates = pd.to_datetime(merged["date"])
    return f"Период тестирования: {dates.min():%Y-%m-%d} - {dates.max():%Y-%m-%d}"


def load_strategy_xlsx(path: Path, pnl_column_name: str) -> pd.DataFrame:
    """Загружает XLSX стратегии, валидирует колонки и нормализует данные по датам."""
    if not path.exists():
        raise typer.BadParameter(f"Файл backtest XLSX не найден: {path}")

    df = pd.read_excel(path)
    required = {"source_date", "pnl"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(
            f"XLSX {path} не содержит обязательные колонки: {missing}. "
            "Ожидаются как минимум source_date и pnl."
        )

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(df["source_date"], errors="coerce").dt.date,
            pnl_column_name: pd.to_numeric(df["pnl"], errors="coerce"),
        }
    )
    return result.dropna(subset=["date", pnl_column_name]).sort_values("date").reset_index(drop=True)


def merge_strategies(gemma_df: pd.DataFrame, qwen_df: pd.DataFrame) -> pd.DataFrame:
    """Объединяет две стратегии по дате и считает простую комбинированную серию P/L."""
    merged = pd.merge(gemma_df, qwen_df, on="date", how="outer").sort_values("date").reset_index(drop=True)
    merged["pnl_gemma"] = merged["pnl_gemma"].fillna(0.0)
    merged["pnl_qwen"] = merged["pnl_qwen"].fillna(0.0)
    merged["pnl_combined"] = (merged["pnl_gemma"] + merged["pnl_qwen"]) / 2.0
    merged["cum_gemma"] = merged["pnl_gemma"].cumsum()
    merged["cum_qwen"] = merged["pnl_qwen"].cumsum()
    merged["cum_combined"] = merged["pnl_combined"].cumsum()
    merged["date_dt"] = pd.to_datetime(merged["date"])
    return merged


def _max_consecutive(signs: pd.Series, target: int) -> int:
    """Возвращает максимальную длину подряд идущих значений, равных target."""
    best = current = 0
    for sign in signs:
        if sign == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _drawdown_duration(drawdown: pd.Series) -> int:
    """Вычисляет максимальную длительность просадки в количестве сделок."""
    max_dd_duration = 0
    current_dd_start = None
    for i, dd_value in enumerate(drawdown):
        if dd_value < 0:
            if current_dd_start is None:
                current_dd_start = i
        else:
            if current_dd_start is not None:
                duration = i - current_dd_start
                if duration > max_dd_duration:
                    max_dd_duration = duration
                current_dd_start = None
    if current_dd_start is not None:
        duration = len(drawdown) - current_dd_start
        if duration > max_dd_duration:
            max_dd_duration = duration
    return max_dd_duration


def calc_stats(pnl: pd.Series, name: str) -> dict:
    """Считает ключевые метрики стратегии для сравнительной таблицы."""
    cum = pnl.cumsum()
    total = float(pnl.sum())
    trades = int((pnl != 0).sum())
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    win_rate = wins / trades * 100 if trades else 0.0
    avg_win = float(pnl[pnl > 0].mean()) if wins else 0.0
    avg_loss = float(pnl[pnl < 0].mean()) if losses else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss else 0.0
    expectancy = float(pnl[pnl != 0].mean()) if trades else 0.0
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = float(drawdown.min())
    pf_gross = float(pnl[pnl > 0].sum())
    pf_loss = abs(float(pnl[pnl < 0].sum()))
    profit_factor = pf_gross / pf_loss if pf_loss else 0.0
    recovery = total / abs(max_dd) if max_dd else 0.0
    return {
        "Стратегия": name,
        "Сделок": trades,
        "Win%": f"{win_rate:.1f}",
        "Total P/L": f"{total:,.0f}",
        "Max DD": f"{max_dd:,.0f}",
        "PF": f"{profit_factor:.2f}",
        "Payoff": f"{payoff:.2f}",
        "Expectancy": f"{expectancy:,.0f}",
        "Recovery": f"{recovery:.2f}",
    }


def build_compare_figure(merged: pd.DataFrame, ticker: str) -> go.Figure:
    """Строит общий график equity для Gemma, Qwen и их комбинации."""
    fig_compare = go.Figure()
    fig_compare.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["cum_gemma"],
            mode="lines",
            line=dict(color="#2e7d32", width=2),
            name="Gemma",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig_compare.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["cum_qwen"],
            mode="lines",
            line=dict(color="#1565c0", width=2),
            name="Qwen",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig_compare.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["cum_combined"],
            mode="lines",
            line=dict(color="#6a1b9a", width=2.5),
            name="Комбинация (среднее)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig_compare.update_layout(
        height=600,
        title_text=f"Сравнение стратегий Gemma vs Qwen — {ticker}",
        title_x=0.5,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        template="plotly_white",
        hovermode="x unified",
    )
    fig_compare.update_yaxes(tickformat=",")
    return fig_compare


def build_combined_report_components(merged: pd.DataFrame, ticker: str) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Строит графики и таблицы для подробного отчёта по комбинированной стратегии."""
    pl = merged["pnl_combined"].astype(float)
    cum = pl.cumsum()
    dates = merged["date_dt"]

    day_colors = ["#d32f2f" if value < 0 else "#2e7d32" for value in pl]

    report_df = merged.copy()
    report_df["Неделя"] = dates.dt.to_period("W")
    weekly = report_df.groupby("Неделя", as_index=False)["pnl_combined"].sum()
    weekly["dt"] = weekly["Неделя"].apply(lambda period: period.start_time)
    week_colors = ["#d32f2f" if value < 0 else "#00838f" for value in weekly["pnl_combined"]]

    report_df["Месяц"] = dates.dt.to_period("M")
    monthly = report_df.groupby("Месяц", as_index=False)["pnl_combined"].sum()
    monthly["dt"] = monthly["Месяц"].dt.to_timestamp()
    month_colors = ["#d32f2f" if value < 0 else "#1565c0" for value in monthly["pnl_combined"]]

    running_max = cum.cummax()
    drawdown = cum - running_max

    total_profit = float(cum.iloc[-1]) if not cum.empty else 0.0
    total_trades = int((pl != 0).sum())
    win_trades = int((pl > 0).sum())
    loss_trades = int((pl < 0).sum())
    win_rate = win_trades / max(total_trades, 1) * 100
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    best_trade = float(pl.max()) if not pl.empty else 0.0
    worst_trade = float(pl.min()) if not pl.empty else 0.0
    avg_trade = float(pl.mean()) if not pl.empty else 0.0
    median_trade = float(pl.median()) if not pl.empty else 0.0
    std_trade = float(pl.std()) if total_trades > 1 else 0.0

    gross_profit = float(pl[pl > 0].sum())
    gross_loss = float(abs(pl[pl < 0].sum()))
    avg_win = float(pl[pl > 0].mean()) if win_trades else 0.0
    avg_loss = float(abs(pl[pl < 0].mean())) if loss_trades else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")
    recovery_factor = total_profit / abs(max_dd) if max_dd != 0 else float("inf")
    expectancy = (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss
    sharpe = (avg_trade / std_trade) * np.sqrt(252) if std_trade > 0 else 0.0

    downside = pl[pl < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = (avg_trade / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

    date_range_days = (dates.max() - dates.min()).days if not dates.empty else 1
    date_range_days = date_range_days or 1
    annual_profit = total_profit * 365 / date_range_days
    calmar = annual_profit / abs(max_dd) if max_dd != 0 else float("inf")

    signs = pl.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    max_consec_wins = _max_consecutive(signs, 1)
    max_consec_losses = _max_consecutive(signs, -1)
    max_dd_duration = _drawdown_duration(drawdown)
    volatility = std_trade * np.sqrt(252)

    stats_text = (
        f"Итого: {total_profit:,.0f} | Сделок: {total_trades} | "
        f"Win: {win_trades} ({win_rate:.0f}%) | Loss: {loss_trades} | "
        f"PF: {profit_factor:.2f} | RF: {recovery_factor:.2f} | "
        f"Sharpe: {sharpe:.2f} | MaxDD: {max_dd:,.0f}"
    )

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "P/L по сделкам",
            "Накопленная прибыль (equity)",
            "P/L по неделям",
            "P/L по месяцам",
            "Drawdown от максимума",
            "Распределение P/L сделок",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "histogram"}],
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Bar(
            x=merged["date"],
            y=pl,
            marker_color=day_colors,
            name="P/L сделки",
            hovertemplate="%{x|%Y-%m-%d}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=cum,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#6a1b9a", width=2),
            fillcolor="rgba(106,27,154,0.15)",
            name="Equity",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=weekly["dt"],
            y=weekly["pnl_combined"],
            marker_color=week_colors,
            name="P/L неделя",
            hovertemplate="Нед. %{x|%Y-%m-%d}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=monthly["dt"],
            y=monthly["pnl_combined"],
            marker_color=month_colors,
            name="P/L месяц",
            text=[f"{value:,.0f}" for value in monthly["pnl_combined"]],
            textposition="outside",
            hovertemplate="%{x|%Y-%m}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#d32f2f", width=1.5),
            fillcolor="rgba(211,47,47,0.2)",
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:,.0f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=pl[pl > 0], marker_color="#2e7d32", opacity=0.7, name="Прибыль", nbinsx=20),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=pl[pl < 0], marker_color="#d32f2f", opacity=0.7, name="Убыток", nbinsx=20),
        row=3,
        col=2,
    )
    fig.update_layout(
        height=1400,
        width=1500,
        title_text=f"Комбинированная стратегия Gemma + Qwen — {ticker}<br><sub>{stats_text}</sub>",
        title_x=0.5,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.03, xanchor="center", x=0.5),
        template="plotly_white",
        hovermode="x unified",
    )
    for row, col in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]:
        fig.update_yaxes(tickformat=",", row=row, col=col)

    sec1 = [
        ["<b>ДОХОДНОСТЬ</b>", ""],
        ["Чистая прибыль", f"{total_profit:,.0f}"],
        ["Годовая прибыль (экстрапол.)", f"{annual_profit:,.0f}"],
        ["Средний P/L на сделку", f"{avg_trade:,.0f}"],
        ["Медианный P/L на сделку", f"{median_trade:,.0f}"],
        ["Лучшая сделка", f"{best_trade:,.0f}"],
        ["Худшая сделка", f"{worst_trade:,.0f}"],
    ]
    sec2 = [
        ["<b>РИСК</b>", ""],
        ["Max Drawdown", f"{max_dd:,.0f}"],
        ["Длит. макс. просадки", f"{max_dd_duration} сделок"],
        ["Волатильность (год.)", f"{volatility:,.0f}"],
        ["Std сделки", f"{std_trade:,.0f}"],
        ["VaR 95%", f"{np.percentile(pl, 5):,.0f}"],
        ["CVaR 95%", f"{pl[pl <= np.percentile(pl, 5)].mean():,.0f}"],
    ]
    sec3 = [
        ["<b>СТАТИСТИКА СДЕЛОК</b>", ""],
        ["Всего сделок", f"{total_trades}"],
        ["Win / Loss", f"{win_trades} / {loss_trades}"],
        ["Win rate", f"{win_rate:.1f}%"],
        ["Ср. выигрыш / проигрыш", f"{avg_win:,.0f} / {avg_loss:,.0f}"],
        ["Макс. серия побед", f"{max_consec_wins}"],
        ["Макс. серия убытков", f"{max_consec_losses}"],
    ]

    num_rows = max(len(sec1), len(sec2), len(sec3))
    for sec in (sec1, sec2, sec3):
        while len(sec) < num_rows:
            sec.append(["", ""])

    cols_values = [[], [], [], [], [], []]
    tbl_colors = [[], [], []]
    for i in range(num_rows):
        for j, sec in enumerate((sec1, sec2, sec3)):
            name, value = sec[i]
            is_header = value == "" and name.startswith("<b>")
            cols_values[j * 2].append(name)
            cols_values[j * 2 + 1].append(f"<b>{value}</b>" if value and not is_header else value)
            if is_header:
                tbl_colors[j].append("#e3f2fd")
            else:
                tbl_colors[j].append("#f5f5f5" if i % 2 == 0 else "white")

    fig_stats = go.Figure(
        go.Table(
            columnwidth=[200, 130, 180, 120, 220, 120],
            header=dict(
                values=["<b>Показатель</b>", "<b>Значение</b>"] * 3,
                fill_color="#1565c0",
                font=dict(color="white", size=14),
                align="left",
                height=32,
            ),
            cells=dict(
                values=cols_values,
                fill_color=[tbl_colors[0], tbl_colors[0], tbl_colors[1], tbl_colors[1], tbl_colors[2], tbl_colors[2]],
                font=dict(size=13, color="#212121"),
                align=["left", "right", "left", "right", "left", "right"],
                height=26,
            ),
        )
    )
    fig_stats.update_layout(
        title_text=f"<b>Комбинация Gemma + Qwen — {ticker}: статистика стратегии</b>",
        title_x=0.5,
        title_font_size=18,
        height=32 + num_rows * 26 + 80,
        width=1500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    coefficients = [
        {
            "name": "Recovery Factor",
            "formula": "Чистая прибыль / |Max Drawdown|",
            "value": f"{recovery_factor:.2f}",
            "description": "Коэффициент восстановления — во сколько раз прибыль превышает максимальную просадку. RF > 1 — стратегия заработала больше, чем потеряла в худший период.",
        },
        {
            "name": "Profit Factor",
            "formula": "Валовая прибыль / Валовый убыток",
            "value": f"{profit_factor:.2f}",
            "description": "Фактор прибыли. PF > 1 — прибыльность, 1.5–2.0 хорошо, > 2.0 отлично.",
        },
        {
            "name": "Payoff Ratio",
            "formula": "Средний выигрыш / Средний проигрыш",
            "value": f"{payoff_ratio:.2f}",
            "description": "При высоком payoff стратегия остаётся прибыльной даже при win rate < 50%.",
        },
        {
            "name": "Sharpe Ratio",
            "formula": "(Ср. P/L / Std) × √252",
            "value": f"{sharpe:.2f}",
            "description": "Отношение доходности к риску, приведённое к году. > 1 хорошо, > 2 отлично, > 3 исключительно.",
        },
        {
            "name": "Sortino Ratio",
            "formula": "(Ср. P/L / Downside Std) × √252",
            "value": f"{sortino:.2f}",
            "description": "Вариант Sharpe, учитывающий только отрицательные отклонения.",
        },
        {
            "name": "Calmar Ratio",
            "formula": "Годовая доходность / |Max Drawdown|",
            "value": f"{calmar:.2f}",
            "description": "Отношение годовой прибыли к макс. просадке. > 1 — прибыль превышает худшую просадку, > 3 отлично.",
        },
        {
            "name": "Expectancy",
            "formula": "Win% × Ср.выигрыш − Loss% × Ср.проигрыш",
            "value": f"{expectancy:,.0f}",
            "description": "Матожидание на одну сделку. Положительное — стратегия имеет преимущество (edge).",
        },
    ]

    fig_table = go.Figure(
        go.Table(
            columnwidth=[150, 250, 80, 450],
            header=dict(
                values=["<b>Коэффициент</b>", "<b>Формула</b>", "<b>Значение</b>", "<b>Расшифровка</b>"],
                fill_color="#1565c0",
                font=dict(color="white", size=14),
                align="left",
                height=36,
            ),
            cells=dict(
                values=[
                    [f"<b>{coef['name']}</b>" for coef in coefficients],
                    [coef["formula"] for coef in coefficients],
                    [f"<b>{coef['value']}</b>" for coef in coefficients],
                    [coef["description"] for coef in coefficients],
                ],
                fill_color=[["#f5f5f5" if i % 2 == 0 else "white" for i in range(len(coefficients))]] * 4,
                font=dict(size=13, color="#212121"),
                align=["left", "left", "center", "left"],
                height=60,
            ),
        )
    )
    fig_table.update_layout(
        title_text=f"<b>Комбинация Gemma + Qwen — {ticker}: ключевые коэффициенты</b>",
        title_x=0.5,
        title_font_size=18,
        height=560,
        width=1500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig, fig_stats, fig_table


def build_stats_table_html(merged: pd.DataFrame) -> str:
    """Строит HTML сводной таблицы сравнения Gemma, Qwen и их комбинации."""
    stats = pd.DataFrame(
        [
            calc_stats(merged["pnl_gemma"], "Gemma"),
            calc_stats(merged["pnl_qwen"], "Qwen"),
            calc_stats(merged["pnl_combined"], "Комбинация"),
        ]
    )
    return stats.to_html(index=False, classes="stats-table", border=0)


def write_html_report(
    output_html: Path,
    stats_html: str,
    fig_compare: go.Figure,
    fig: go.Figure,
    fig_stats: go.Figure,
    fig_table: go.Figure,
    browser_title: str,
    page_heading: str,
    test_period: str,
) -> None:
    """Сохраняет итоговый HTML-отчёт со сводной таблицей и интерактивными графиками."""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    escaped_title = escape(browser_title, quote=False)
    escaped_heading = escape(page_heading, quote=False)
    escaped_period = escape(test_period, quote=False)
    with output_html.open("w", encoding="utf-8") as file_obj:
        file_obj.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
        file_obj.write(f"<title>{escaped_title}</title>\n")
        file_obj.write("<style>\n")
        file_obj.write("body { font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }\n")
        file_obj.write(".stats-table { border-collapse: collapse; margin: 20px auto; font-size: 14px; }\n")
        file_obj.write(".stats-table th { background: #37474f; color: white; padding: 8px 16px; }\n")
        file_obj.write(".stats-table td { padding: 6px 16px; border-bottom: 1px solid #ddd; text-align: center; }\n")
        file_obj.write(".stats-table tr:hover { background: #e3f2fd; }\n")
        file_obj.write("</style>\n</head><body>\n")
        file_obj.write(f"<h2 style='text-align:center;'>{escaped_heading}</h2>\n")
        file_obj.write(f"<p style='text-align:center; margin-top:-8px;'>{escaped_period}</p>\n")
        file_obj.write(stats_html)
        file_obj.write(fig_compare.to_html(include_plotlyjs="cdn", full_html=False))
        file_obj.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        file_obj.write(fig.to_html(include_plotlyjs=False, full_html=False))
        file_obj.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        file_obj.write(fig_stats.to_html(include_plotlyjs=False, full_html=False))
        file_obj.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        file_obj.write(fig_table.to_html(include_plotlyjs=False, full_html=False))
        file_obj.write("\n</body></html>")


@app.command()
def main(
    gemma_xlsx: Path = typer.Option(
        GEMMA_XLSX,
        help="XLSX с результатами backtest стратегии Gemma.",
    ),
    qwen_xlsx: Path = typer.Option(
        QWEN_XLSX,
        help="XLSX с результатами backtest стратегии Qwen.",
    ),
    output_html: Path = typer.Option(
        OUTPUT_HTML,
        help="HTML-файл для сохранения сравнительного отчёта.",
    ),
) -> None:
    """Запускает сравнение двух sentiment-стратегий и сохраняет HTML-отчёт."""
    gemma_df = load_strategy_xlsx(gemma_xlsx, "pnl_gemma")
    qwen_df = load_strategy_xlsx(qwen_xlsx, "pnl_qwen")
    merged = merge_strategies(gemma_df, qwen_df)

    if merged.empty:
        typer.echo("Нет данных для сравнения. Проверьте входные XLSX.")
        raise typer.Exit(code=1)

    stats_html = build_stats_table_html(merged)
    labels = load_report_labels()
    ticker = labels["ticker"]
    fig_compare = build_compare_figure(merged, ticker)
    fig, fig_stats, fig_table = build_combined_report_components(merged, ticker)
    write_html_report(
        output_html,
        stats_html,
        fig_compare,
        fig,
        fig_stats,
        fig_table,
        build_browser_title(),
        build_page_heading(),
        build_test_period(merged),
    )

    typer.echo(f"Отчёт сохранён: {output_html}")


if __name__ == "__main__":
    app()
