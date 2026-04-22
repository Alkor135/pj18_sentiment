"""Открывает HTML-отчёты текущего RTS sentiment pipeline в Google Chrome.

Скрипт работает только с актуальными отчётами этого пайплайна:
- `rts/sentiment_gemma/plots/sentiment_backtest.html`
- `rts/sentiment_gemma/plots/sentiment_backtest_qs.html`
- `rts/sentiment_qwen/plots/sentiment_backtest.html`
- `rts/sentiment_qwen/plots/sentiment_backtest_qs.html`
- `rts/combine/plots/sentiment_compare.html`
- `mix/sentiment_gemma/plots/sentiment_backtest.html`
- `mix/sentiment_gemma/plots/sentiment_backtest_qs.html`
- `mix/sentiment_qwen/plots/sentiment_backtest.html`
- `mix/sentiment_qwen/plots/sentiment_backtest_qs.html`
- `mix/combine/plots/sentiment_compare.html`

Открывает найденные файлы в одном новом окне Chrome в фиксированном порядке.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHROME_PATH = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
HTML_REPORTS = [
    ROOT / "rts" / "sentiment_gemma" / "plots" / "sentiment_backtest.html",
    ROOT / "rts" / "sentiment_gemma" / "plots" / "sentiment_backtest_qs.html",
    ROOT / "rts" / "sentiment_qwen" / "plots" / "sentiment_backtest.html",
    ROOT / "rts" / "sentiment_qwen" / "plots" / "sentiment_backtest_qs.html",
    ROOT / "rts" / "combine" / "plots" / "sentiment_compare.html",
    ROOT / "mix" / "sentiment_gemma" / "plots" / "sentiment_backtest.html",
    ROOT / "mix" / "sentiment_gemma" / "plots" / "sentiment_backtest_qs.html",
    ROOT / "mix" / "sentiment_qwen" / "plots" / "sentiment_backtest.html",
    ROOT / "mix" / "sentiment_qwen" / "plots" / "sentiment_backtest_qs.html",
    ROOT / "mix" / "combine" / "plots" / "sentiment_compare.html",
]


def collect_existing_reports(report_paths: list[Path]) -> list[Path]:
    """Возвращает только существующие HTML-отчёты, сохраняя заданный порядок."""
    return [path for path in report_paths if path.exists()]


def open_reports_in_chrome(chrome_path: Path, report_paths: list[Path]) -> None:
    """Открывает список HTML-отчётов в одном новом окне Google Chrome."""
    if not chrome_path.exists():
        print(f"Google Chrome не найден: {chrome_path}")
        raise SystemExit(1)

    subprocess.Popen([str(chrome_path), "--new-window", *[str(path) for path in report_paths]])


def main() -> None:
    """Собирает HTML-отчёты pipeline и открывает их в браузере."""
    files = collect_existing_reports(HTML_REPORTS)

    if not files:
        print("HTML-отчёты текущего RTS sentiment pipeline не найдены.")
        raise SystemExit(0)

    open_reports_in_chrome(CHROME_PATH, files)

    print("Открываю HTML-отчёты:")
    for path in files:
        print(f"[ОТКРЫВАЮ] {path}")


if __name__ == "__main__":
    main()
