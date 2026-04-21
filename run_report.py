"""
Оркестратор отчётного пайплайна pj18_sentiment.

Скрипт запускает подготовленные этапы анализа настроений для инструментов RTS и MIX
и пишет подробный лог в каталог `log/`. Предполагается, что входные данные для
отчёта уже подготовлены предыдущими этапами: минутные и дневные данные, а также
новостные Markdown-файлы.

Шаги из `HARD_STEPS` считаются обязательными: отсутствие скрипта, исключение или
ненулевой код возврата останавливают весь пайплайн. Шаги из `SOFT_STEPS`
используются для дополнительной аналитики: ошибки фиксируются в логе, но не
прерывают выполнение.
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
log_file = LOG_DIR / f"run_report_{timestamp}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=build_handlers(log_file),
    force=True,
)
logger = logging.getLogger("run_report")

for old in sorted(LOG_DIR.glob("run_report_*.txt"))[:-3]:
    try:
        old.unlink()
    except Exception:
        pass


HARD_STEPS: list[Path] = [
    # Этап 4: анализ настроения (тяжёлый — Ollama)
    ROOT / "rts" / "sentiment_gemma" / "sentiment_analysis_gemma.py",
    ROOT / "rts" / "sentiment_qwen" / "sentiment_analysis_qwen.py",
    ROOT / "mix" / "sentiment_gemma" / "sentiment_analysis_gemma.py",
    ROOT / "mix" / "sentiment_qwen" / "sentiment_analysis_qwen.py",
    # ROOT / "br" / "embedding" / "create_embedding.py",
    # ROOT / "gold" / "embedding" / "create_embedding.py",
    # ROOT / "ng" / "embedding" / "create_embedding.py",
    # ROOT / "si" / "embedding" / "create_embedding.py",
    # ROOT / "spyf" / "embedding" / "create_embedding.py",

    # Этап 5: групповые статистики
    ROOT / "rts" / "sentiment_gemma" / "sentiment_group_stats.py",
    ROOT / "rts" / "sentiment_qwen" / "sentiment_group_stats.py",
    ROOT / "mix" / "sentiment_gemma" / "sentiment_group_stats.py",
    ROOT / "mix" / "sentiment_qwen" / "sentiment_group_stats.py",
    # ROOT / "br" / "embedding" / "embedding_backtest.py",
    # ROOT / "gold" / "embedding" / "embedding_backtest.py",
    # ROOT / "ng" / "embedding" / "embedding_backtest.py",
    # ROOT / "si" / "embedding" / "embedding_backtest.py",
    # ROOT / "spyf" / "embedding" / "embedding_backtest.py",

    # # Этап 6: бэктест и анализ результатов
    ROOT / "rts" / "sentiment_gemma" / "sentiment_backtest.py",
    ROOT / "rts" / "sentiment_qwen" / "sentiment_backtest.py",
    ROOT / "mix" / "sentiment_gemma" / "sentiment_backtest.py",
    ROOT / "mix" / "sentiment_qwen" / "sentiment_backtest.py",
    # ROOT / "br" / "embedding" / "embedding_to_predict.py",
    # ROOT / "gold" / "embedding" / "embedding_to_predict.py",
    # ROOT / "ng" / "embedding" / "embedding_to_predict.py",
    # ROOT / "si" / "embedding" / "embedding_to_predict.py",
    # ROOT / "spyf" / "embedding" / "embedding_to_predict.py",

    # Этап 7: sentiment-анализ через LLM (тяжёлый — Ollama)
    # ROOT / "rts" / "sentiment" / "sentiment_analysis.py",
    # ROOT / "mix" / "sentiment" / "sentiment_analysis.py",
    # ROOT / "br" / "sentiment" / "sentiment_analysis.py",
    # ROOT / "gold" / "sentiment" / "sentiment_analysis.py",
    # ROOT / "ng" / "sentiment" / "sentiment_analysis.py",
    # ROOT / "si" / "sentiment" / "sentiment_analysis.py",
    # ROOT / "spyf" / "sentiment" / "sentiment_analysis.py",

    # # Этап 8: прогноз sentiment на сегодня (по rules.yaml)
    # ROOT / "rts" / "sentiment" / "sentiment_to_predict.py",
    # ROOT / "mix" / "sentiment" / "sentiment_to_predict.py",
    # ROOT / "br" / "sentiment" / "sentiment_to_predict.py",
    # ROOT / "gold" / "sentiment" / "sentiment_to_predict.py",
    # ROOT / "ng" / "sentiment" / "sentiment_to_predict.py",
    # ROOT / "si" / "sentiment" / "sentiment_to_predict.py",
    # ROOT / "spyf" / "sentiment" / "sentiment_to_predict.py",

    # # Этап 9: согласованное голосование → combined-прогноз
    # ROOT / "rts" / "combine_predictions.py",
    # ROOT / "mix" / "combine_predictions.py",
    # ROOT / "br" / "combine_predictions.py",
    # ROOT / "gold" / "combine_predictions.py",
    # ROOT / "ng" / "combine_predictions.py",
    # ROOT / "si" / "combine_predictions.py",
    # ROOT / "spyf" / "combine_predictions.py",
]

SOFT_STEPS: list[Path] = [
    ROOT / "rts" / "combine" / "sentiment_compare.py",
    ROOT / "mix" / "combine" / "sentiment_compare.py",
    # ROOT / "br" / "embedding" / "embedding_analysis.py",
    # ROOT / "gold" / "embedding" / "embedding_analysis.py",
    # ROOT / "ng" / "embedding" / "embedding_analysis.py",
    # ROOT / "si" / "embedding" / "embedding_analysis.py",
    # ROOT / "spyf" / "embedding" / "embedding_analysis.py",

    # ROOT / "rts" / "sentiment" / "sentiment_group_stats.py",
    # ROOT / "mix" / "sentiment" / "sentiment_group_stats.py",
    # ROOT / "br" / "sentiment" / "sentiment_group_stats.py",
    # ROOT / "gold" / "sentiment" / "sentiment_group_stats.py",
    # ROOT / "ng" / "sentiment" / "sentiment_group_stats.py",
    # ROOT / "si" / "sentiment" / "sentiment_group_stats.py",
    # ROOT / "spyf" / "sentiment" / "sentiment_group_stats.py",

    # ROOT / "rts" / "sentiment" / "sentiment_backtest.py",
    # ROOT / "mix" / "sentiment" / "sentiment_backtest.py",
    # ROOT / "br" / "sentiment" / "sentiment_backtest.py",
    # ROOT / "gold" / "sentiment" / "sentiment_backtest.py",
    # ROOT / "ng" / "sentiment" / "sentiment_backtest.py",
    # ROOT / "si" / "sentiment" / "sentiment_backtest.py",
    # ROOT / "spyf" / "sentiment" / "sentiment_backtest.py",

    # ROOT / "rts" / "sentiment" / "sentiment_compare.py",
    # ROOT / "mix" / "sentiment" / "sentiment_compare.py",
    # ROOT / "br" / "sentiment" / "sentiment_compare.py",
    # ROOT / "gold" / "sentiment" / "sentiment_compare.py",
    # ROOT / "ng" / "sentiment" / "sentiment_compare.py",
    # ROOT / "si" / "sentiment" / "sentiment_compare.py",
    # ROOT / "spyf" / "sentiment" / "sentiment_compare.py",  # последний
]


def run(script: Path, hard: bool) -> int:
    """Запускает один дочерний Python-скрипт и обрабатывает его результат.

    Args:
        script: Абсолютный путь к запускаемому скрипту.
        hard: Если `True`, любая ошибка завершает весь пайплайн через `sys.exit`.
            Если `False`, ошибка логируется как soft-fail, а выполнение продолжается.

    Returns:
        Код возврата дочернего процесса или внутренний код ошибки оркестратора.
    """
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
    """Выполняет отчётный пайплайн: обязательные шаги, затем soft-аналитику.

    Returns:
        `0`, если все обязательные шаги выполнены успешно. При hard-fail функция
        завершает процесс раньше через `run`.
    """
    logger.info(f"=== pj18 run_report.py начат: {timestamp} ===")
    logger.info(f"Python: {sys.executable}")
    logger.info(f"ROOT: {ROOT}")

    for step in HARD_STEPS:
        run(step, hard=True)

    logger.info("--- Торговля завершена, переходим к аналитике (soft-fail) ---")

    for step in SOFT_STEPS:
        run(step, hard=False)

    logger.info("=== pj18 run_report.py завершён успешно ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
