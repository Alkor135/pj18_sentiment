"""
Оркестратор pj18 для ежедневного запуска из Windows Task Scheduler в 21:00:05.

Обрабатывает два тикера (RTS и MIX). Каждый этап выполняется парой RTS → MIX,
чтобы оба .tri попали в QUIK с минимальным зазором между ними:
  0) prepare.py (удаляет тестовые результаты, если запуск до 21:00)
  1) beget/sync_files.py
  2) shared: download_minutes_to_db → convert_minutes_to_days (RTS, MIX); create_markdown_files (только RTS — md_path общий)
  3) embedding: create_embedding → embedding_backtest → embedding_to_predict (инверсия) (RTS, MIX)
  4) sentiment: sentiment_analysis → sentiment_to_predict                                (RTS, MIX)
  5) combine_predictions.py — согласованное голосование                                  (RTS, MIX)
  6) trade/trade_rts_sentiment_SPBFUT192yc_ebs.py  +  trade/trade_mix_sentiment_SPBFUT192yc_ebs.py  ← встык, критично по времени
  7) Аналитика (soft-fail): embedding_analysis, sentiment_group_stats, sentiment_backtest,
     sentiment_compare — по обоим тикерам; sentiment_compare MIX идёт последним.

Hard-fail (exit с кодом ошибки) до и включая trade-скрипты — чтобы при сбое на торговом
этапе сразу поднять алерт. После trade-скриптов ошибки — warning, пайплайн продолжается.

Регистрация в планировщике Windows:
  schtasks /Create /SC DAILY /ST 21:00:05 /TN "pj18_run_all" ^
      /TR "C:\\Users\\Alkor\\VSCode\\pj18_sentiment\\.venv\\Scripts\\python.exe C:\\Users\\Alkor\\VSCode\\pj18_sentiment\\run_all.py"
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from orchestrator_logging import build_handlers

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = LOG_DIR / f"run_all_{timestamp}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=build_handlers(log_file),
    force=True,
)
logger = logging.getLogger("run_all")

for old in sorted(LOG_DIR.glob("run_all_*.txt"))[:-3]:
    try:
        old.unlink()
    except Exception:
        pass


HARD_STEPS: list[Path] = [
    ROOT / "prepare.py",  # удаление тестовых файлов, если запуск до 21:00:00 (защита рабочих результатов)
    ROOT / "beget" / "sync_files.py",  # синхронизация файлов с удалённого сервера (включая .tri для QUIK)

    # Этап 1: загрузка минутных котировок в БД (у каждого тикера своя БД)
    ROOT / "rts" / "shared" / "download_minutes_to_db.py",
    ROOT / "mix" / "shared" / "download_minutes_to_db.py",

    # Этап 2: агрегация минут → дневные свечи
    ROOT / "rts" / "shared" / "convert_minutes_to_days.py",
    ROOT / "mix" / "shared" / "convert_minutes_to_days.py",

    # Этап 3: markdown-сводки новостей по торговым сессиям (md_path общий, второй вызов избыточен)
    ROOT / "rts" / "shared" / "create_markdown_files.py",

    # Этап 4: sentiment-анализ по первой модели (тяжёлый — Ollama)
    ROOT / "rts" / "sentiment_gemma" / "sentiment_analysis_gemma.py",
    ROOT / "mix" / "sentiment_gemma" / "sentiment_analysis_gemma.py",

    # Этап 5: sentiment-анализ по второй модели (тяжёлый — Ollama)
    ROOT / "rts" / "sentiment_qwen" / "sentiment_analysis_qwen.py",
    ROOT / "mix" / "sentiment_qwen" / "sentiment_analysis_qwen.py",

    # Этап 6: создание файлов с прогнозами на сегодня по rules.yaml (по каждой модели и тикеру отдельно)
    ROOT / "rts" / "sentiment_gemma" / "sentiment_to_predict.py",
    ROOT / "mix" / "sentiment_gemma" / "sentiment_to_predict.py",
    ROOT / "rts" / "sentiment_qwen" / "sentiment_to_predict.py",
    ROOT / "mix" / "sentiment_qwen" / "sentiment_to_predict.py",

    # # Этап 7: sentiment-анализ через LLM (тяжёлый — Ollama)
    # ROOT / "rts" / "sentiment" / "sentiment_analysis.py",
    # ROOT / "mix" / "sentiment" / "sentiment_analysis.py",

    # # Этап 8: прогноз sentiment на сегодня (по rules.yaml)
    # ROOT / "rts" / "sentiment" / "sentiment_to_predict.py",
    # ROOT / "mix" / "sentiment" / "sentiment_to_predict.py",

    # # Этап 9: согласованное голосование → combined-прогноз
    # ROOT / "rts" / "combine_predictions.py",
    # ROOT / "mix" / "combine_predictions.py",

    # Этап 10: торговые скрипты встык — оба .tri попадают в QUIK с минимальным лагом
    # RTS и MIX временно торгуются по sentiment-стратегии (combo-скрипты отключены):
    # ROOT / "trade" / "trade_rts_combo_SPBFUT192yc_ebs.py",
    # ROOT / "trade" / "trade_mix_combo_SPBFUT192yc_ebs.py",
    # ROOT / "trade" / "trade_rts_sentiment_SPBFUT192yc_ebs.py",
    # ROOT / "trade" / "trade_mix_sentiment_SPBFUT192yc_ebs.py",
]

SOFT_STEPS: list[Path] = [
    ROOT / "rts" / "sentiment_gemma" / "sentiment_group_stats.py",
    ROOT / "mix" / "sentiment_gemma" / "sentiment_group_stats.py",

    ROOT / "rts" / "sentiment_qwen" / "sentiment_group_stats.py",
    ROOT / "mix" / "sentiment_qwen" / "sentiment_group_stats.py",

    ROOT / "rts" / "sentiment_gemma" / "sentiment_backtest.py",
    ROOT / "mix" / "sentiment_gemma" / "sentiment_backtest.py",

    ROOT / "rts" / "sentiment_qwen" / "sentiment_backtest.py",
    ROOT / "mix" / "sentiment_qwen" / "sentiment_backtest.py",
]


def run(script: Path, hard: bool) -> int:
    if not script.exists():
        msg = f"СКРИПТ НЕ НАЙДЕН: {script}"
        logger.error(msg)
        if hard:
            sys.exit(2)
        logger.warning(msg)
        return 2

    logger.info(f"▶ {'HARD' if hard else 'soft'}: {script.relative_to(ROOT)}")
    start = datetime.now()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            check=False,
        )
        rc = proc.returncode
    except Exception as exc:
        logger.error(f"Исключение при запуске {script.name}: {exc}")
        if hard:
            sys.exit(3)
        return 3

    elapsed = (datetime.now() - start).total_seconds()
    if rc == 0:
        logger.info(f"✓ {script.name} — OK ({elapsed:.1f} сек)")
    else:
        if hard:
            logger.error(
                f"✗ {script.name} упал с кодом {rc} ({elapsed:.1f} сек). Останов пайплайна."
            )
            sys.exit(rc)
        logger.warning(
            f"⚠ {script.name} упал с кодом {rc} ({elapsed:.1f} сек). Продолжаем (soft-fail)."
        )
    return rc


def main() -> int:
    logger.info(f"=== pj18 run_all.py начат: {timestamp} ===")
    logger.info(f"Python: {sys.executable}")
    logger.info(f"ROOT: {ROOT}")

    for step in HARD_STEPS:
        run(step, hard=True)

    logger.info("--- Торговля завершена, переходим к аналитике (soft-fail) ---")

    for step in SOFT_STEPS:
        run(step, hard=False)

    logger.info("=== pj18 run_all.py завершён успешно ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
