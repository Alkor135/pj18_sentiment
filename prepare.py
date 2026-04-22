"""
Подготовка к тестовому запуску пайплайна.

Если скрипт запущен до 21:00:00 текущего дня — удаляет сегодняшние результаты,
чтобы пайплайн можно было прогнать заново:
  - Файлы предсказаний за сегодня (gemma, qwen, combined) по RTS/MIX
  - Done-маркеры за сегодня в trade/state/*.done (защита от повторной записи)

Если скрипт запущен после 21:00:00 — ничего сегодняшнего не трогает, чтобы
защитить рабочие результаты официального ночного запуска (21:00:05).

В любом случае (до и после 21:00:00) выполняется housekeeping:
  - trade/state/*.done: хранится не более DONE_RETENTION_DAYS календарных дней
    и не более DONE_RETENTION_FILES файлов (см. константы ниже).
  - log/prepare_*.txt: оставляется 3 самых свежих лога этого скрипта.

Замечание по импорту: настройка logging (в т.ч. создание файла log/prepare_*.txt)
сделана внутри main(), чтобы `import prepare` в тестах не плодил пустые логи.

Используется в начале run_all.py как первый шаг (перед основным пайплайном).
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = ROOT / "trade" / "state"
DONE_RETENTION_DAYS = 10
DONE_RETENTION_FILES = 10

logger = logging.getLogger("prepare")


PREDICT_SECTIONS = ("sentiment_gemma", "sentiment_qwen", "combined")


def parse_done_marker_date(path: Path) -> date | None:
    """Извлекает дату из имени .done-маркера формата <prefix>_YYYY-MM-DD.done."""
    if path.suffix != ".done":
        return None

    stem = path.stem
    prefix, sep, date_str = stem.rpartition("_")
    if not prefix or not sep:
        return None

    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def get_done_markers_to_delete(
    marker_paths,
    *,
    today: date,
    max_age_days: int = DONE_RETENTION_DAYS,
    max_files: int = DONE_RETENTION_FILES,
) -> list[Path]:
    """
    Возвращает .done-маркеры для удаления:
      - старше max_age_days дней от today
      - сверх лимита max_files среди оставшихся валидных маркеров
    """
    dated_markers: list[tuple[Path, date]] = []
    to_delete: list[Path] = []

    for marker_path in marker_paths:
        marker_date = parse_done_marker_date(marker_path)
        if marker_date is None:
            continue
        dated_markers.append((marker_path, marker_date))

    min_date = today - timedelta(days=max_age_days - 1)
    kept_candidates: list[tuple[Path, date]] = []

    for marker_path, marker_date in dated_markers:
        if marker_date < min_date:
            to_delete.append(marker_path)
        else:
            kept_candidates.append((marker_path, marker_date))

    kept_candidates.sort(key=lambda item: (item[1], item[0].name), reverse=True)
    overflow = kept_candidates[max_files:]
    to_delete.extend(path for path, _ in overflow)

    return sorted(to_delete, key=lambda path: path.name, reverse=True)


def load_yaml(path: Path) -> dict:
    """Читает YAML-файл как dict."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def merge_settings_section(raw: dict, section: str) -> dict:
    """Объединяет common с указанной секцией и подставляет ticker-плейсхолдеры."""
    settings = {**(raw.get("common") or {}), **(raw.get(section) or {})}
    ticker = settings.get("ticker", "")
    ticker_lc = settings.get("ticker_lc", str(ticker).lower())
    for key, value in list(settings.items()):
        if isinstance(value, str):
            settings[key] = value.replace("{ticker}", ticker).replace("{ticker_lc}", ticker_lc)
    return settings


def collect_predict_files_for_date(
    root: Path,
    target_date: date,
    tickers: tuple[str, ...] = ("rts", "mix"),
    sections: tuple[str, ...] = PREDICT_SECTIONS,
) -> list[Path]:
    """Возвращает файлы прогнозов за дату по predict_path из settings.yaml."""
    date_filename = target_date.strftime("%Y-%m-%d") + ".txt"
    files: list[Path] = []

    for ticker in tickers:
        raw_settings = load_yaml(root / ticker / "settings.yaml")
        for section in sections:
            settings = merge_settings_section(raw_settings, section)
            predict_path = settings.get("predict_path")
            if predict_path:
                files.append(Path(predict_path) / date_filename)

    return files


def cleanup_prepare_logs(log_dir: Path, max_files: int = 3) -> None:
    """Оставляет только max_files самых новых логов prepare."""
    for old in sorted(log_dir.glob("prepare_*.txt"))[:-max_files]:
        try:
            old.unlink()
        except Exception:
            pass


def main() -> int:
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    hour = now.hour
    minute = now.minute

    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"prepare_{timestamp}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logger.info(f"=== prepare.py начат: {timestamp} ===")
    logger.info(f"Текущее время: {hour:02d}:{minute:02d}, дата: {today_str}")
    cleanup_prepare_logs(LOG_DIR)

    # Порог: 21:00:00 (21 час, начиная с минуты 0)
    cutoff_hour = 21
    is_before_cutoff = hour < cutoff_hour

    done_markers = list(STATE_DIR.glob("*.done"))
    old_done_markers = get_done_markers_to_delete(
        done_markers,
        today=now.date(),
        max_age_days=DONE_RETENTION_DAYS,
        max_files=DONE_RETENTION_FILES,
    )
    if old_done_markers:
        logger.info(
            "Очистка истории .done: удаляем маркеры старше "
            f"{DONE_RETENTION_DAYS} дней и сверх лимита {DONE_RETENTION_FILES} файлов."
        )

    deleted_count = 0
    for filepath in old_done_markers:
        if filepath.exists():
            try:
                filepath.unlink()
                logger.info(f"  Удалён старый done-маркер: {filepath}")
                deleted_count += 1
            except Exception as exc:
                logger.warning(f"  Не удалось удалить {filepath}: {exc}")

    if not is_before_cutoff:
        logger.info(
            f"Текущее время >= 21:00:00 — это рабочее время. "
            f"Тестовые файлы не удаляются, защита рабочих результатов.\n"
        )
        return 0

    logger.info(f"Текущее время < 21:00:00 — дневное тестирование. Удаляем результаты за {today_str}...")

    files_to_delete = collect_predict_files_for_date(ROOT, now.date())

    today_done_markers = [
        path for path in done_markers if parse_done_marker_date(path) == now.date()
    ]

    all_files = files_to_delete + today_done_markers

    for filepath in all_files:
        if filepath.exists():
            try:
                filepath.unlink()
                logger.info(f"  Удалён: {filepath}")
                deleted_count += 1
            except Exception as exc:
                logger.warning(f"  Не удалось удалить {filepath}: {exc}")

    logger.info(f"\nУдалено файлов: {deleted_count}")
    logger.info("=== prepare.py завершён успешно ===\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
