"""
Генерирует файл предсказания направления цены на текущую торговую дату.

Читает sentiment_scores.pkl, берёт строку за сегодня (одна дата — одна строка),
применяет правила из rules.yaml и пишет текстовый файл <predict_path>/YYYY-MM-DD.txt
в формате:

    Дата: 2026-04-09
    Sentiment: -4.00
    Action: invert
    Status: ok
    Предсказанное направление: up

Файл пишется ВСЕГДА (включая нештатные ситуации) — чтобы утром всегда было что
проверить. Направление = skip во всех не-ok случаях; причина — в поле Status,
подробности (если есть) — в поле Note.

Возможные значения Status:
  ok              — нормальная запись направления (up/down)
  no_rule_match   — sentiment не попал ни в один диапазон rules.yaml
  no_pkl_row      — в sentiment_scores.pkl нет строки за сегодня
  pkl_missing     — файл sentiment_scores.pkl не найден
  pkl_duplicate   — в pkl несколько строк за сегодня (ожидалась одна)
  error           — необработанное исключение (traceback краткий — в Note)

Если файл за сегодня уже есть и создан после time_start — не перезаписываем;
если создан до time_start (тестовый прогон) — перезаписывается.

Скрипт всегда возвращает 0, чтобы сбой sentiment по одному тикеру не останавливал
run_all.py и не блокировал обработку остальных тикеров.
"""

from __future__ import annotations

import logging
import pickle
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

TICKER_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


VALID_ACTIONS = {"follow", "invert"}


def cleanup_old_logs(log_dir: Path, max_files: int = 3) -> None:
    """Оставляет только последние max_files логов sentiment_to_predict.

    Удаляет самые старые файлы по времени изменения. Ошибки удаления не
    прерывают основной сценарий, потому что чистка логов не должна мешать
    созданию файла предсказания.
    """
    log_files = sorted(
        log_dir.glob("sentiment_to_predict_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in log_files[max_files:]:
        try:
            old.unlink()
        except Exception as exc:
            print(f"Не удалось удалить старый лог {old}: {exc}")


def setup_logging() -> logging.Logger:
    """Настраивает логирование в новый файл и в консоль.

    Каждый запуск получает отдельный лог с timestamp в имени. После настройки
    удаляются старые логи сверх лимита, чтобы каталог не разрастался.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"sentiment_to_predict_{timestamp}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    cleanup_old_logs(LOG_DIR)
    return logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    """Читает YAML-файл как dict.

    Если YAML пустой, возвращает пустой словарь. Ошибки чтения и парсинга
    намеренно не скрываются: вызывающий код должен записать status=error.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(path: Path, section: str = "sentiment_qwen") -> dict:
    """Загружает настройки инструмента из common и указанной sentiment-секции.

    Специализированная секция перекрывает одноименные ключи из common. После объединения
    подставляет плейсхолдеры {ticker} и {ticker_lc} во все строковые значения.
    """
    raw = load_yaml(path)
    settings = {**(raw.get("common") or {}), **(raw.get(section) or {})}
    ticker = settings.get("ticker", "")
    ticker_lc = settings.get("ticker_lc", str(ticker).lower())
    for key, value in list(settings.items()):
        if isinstance(value, str):
            settings[key] = value.replace("{ticker}", ticker).replace("{ticker_lc}", ticker_lc)
    return settings


def resolve_sentiment_pkl(settings: dict, base_dir: Path = TICKER_DIR) -> Path:
    """Возвращает путь к PKL с учетом суффикса модели.

    Скрипт sentiment-анализа сохраняет файл не только по базовому имени из
    настроек, но и с безопасным суффиксом модели. Относительный путь считается
    от директории инструмента.
    """
    sentiment_path = Path(settings.get("sentiment_output_pkl", "sentiment_scores.pkl"))
    model = str(settings.get("sentiment_model", "model"))
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"
    sentiment_path = sentiment_path.with_name(
        f"{sentiment_path.stem}_{model_slug}{sentiment_path.suffix}"
    )
    return sentiment_path if sentiment_path.is_absolute() else base_dir / sentiment_path


def load_rules(path: Path) -> list[dict]:
    """Загружает и валидирует rules.yaml для прогноза.

    Ожидает список rules, где у каждого правила есть min, max и action.
    В текущем формате для этого скрипта допустимы только follow и invert:
    action=skip больше не используется в rules.yaml.
    """
    data = load_yaml(path)
    rules = data.get("rules") or []
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"В {path} нет списка 'rules' или он пустой")
    for i, rule in enumerate(rules):
        for key in ("min", "max", "action"):
            if key not in rule:
                raise ValueError(f"Правило #{i} без поля '{key}': {rule}")
        if rule["action"] not in VALID_ACTIONS:
            raise ValueError(
                f"Правило #{i}: action должен быть одним из {sorted(VALID_ACTIONS)}, "
                f"получено {rule['action']!r}"
            )
        if float(rule["min"]) > float(rule["max"]):
            raise ValueError(f"Правило #{i}: min > max ({rule})")
    return rules


def match_action(sentiment: float, rules: list[dict]) -> str | None:
    """Возвращает action первого подходящего правила.

    Правила проверяются сверху вниз. Если sentiment не попал ни в один диапазон,
    возвращается None, а main записывает status=no_rule_match и direction=skip.
    """
    for rule in rules:
        if float(rule["min"]) <= sentiment <= float(rule["max"]):
            return rule["action"]
    return None


def resolve_direction(sentiment: float, action: str) -> str:
    """Преобразует sentiment и action в направление up/down.

    follow идет по знаку sentiment: 0 считается положительным и дает up.
    invert инвертирует знак: 0 считается положительным и дает down.
    Неизвестный action возвращает skip как защитный fallback.
    """
    if action == "follow":
        return "up" if sentiment >= 0 else "down"
    if action == "invert":
        return "down" if sentiment >= 0 else "up"
    return "skip"


def get_today_sentiment(pkl_path: Path, today: date) -> float | None:
    """Возвращает sentiment за указанную дату из PKL.

    В PKL должна быть одна строка на дату. Если строки за today нет, возвращает
    None. Если файл отсутствует, кидает FileNotFoundError. Если за дату найдено
    несколько строк или не хватает обязательных колонок, кидает ValueError.
    """
    if not pkl_path.exists():
        raise FileNotFoundError(f"Файл sentiment PKL не найден: {pkl_path}")
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    if "source_date" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            f"PKL не содержит обязательные колонки 'source_date'/'sentiment': {pkl_path}"
        )

    df["source_date"] = pd.to_datetime(df["source_date"], errors="coerce").dt.date
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["source_date", "sentiment"])

    today_rows = df[df["source_date"] == today]
    if today_rows.empty:
        return None
    if len(today_rows) > 1:
        raise ValueError(
            f"В pkl несколько строк за {today}: ожидалась одна. "
            "Перегенерируй pkl: sentiment_analysis.py хранит одну строку на дату."
        )
    return float(today_rows["sentiment"].iloc[0])


def should_rewrite_existing_predict(out_file: Path, today: date, time_start: str) -> bool:
    """Проверяет, нужно ли перезаписать существующий файл прогноза.

    Сравнивает время последнего изменения файла с сегодняшним time_start.
    Если файл изменен до time_start, считаем его тестовым/устаревшим и
    разрешаем перезапись. Если после time_start — не трогаем.
    """
    cutoff = datetime.combine(today, datetime.strptime(time_start, "%H:%M:%S").time())
    file_mtime = datetime.fromtimestamp(out_file.stat().st_mtime)
    return file_mtime < cutoff


def write_predict(
    out_file: Path,
    date_str: str,
    direction: str,
    status: str,
    sentiment: float | None = None,
    action: str | None = None,
    note: str = "",
) -> None:
    """Атомарно записывает текстовый файл прогноза.

    Формирует человекочитаемый файл с датой, sentiment, action, status,
    необязательной Note и строкой "Предсказанное направление". Запись идет
    через временный .tmp-файл и replace(), чтобы не оставить частично записанный
    прогноз при сбое.
    """
    sentiment_label = f"{sentiment:.2f}" if sentiment is not None else "n/a"
    action_label = action if action is not None else "n/a"
    lines = [
        f"Дата: {date_str}",
        f"Sentiment: {sentiment_label}",
        f"Action: {action_label}",
        f"Status: {status}",
    ]
    if note:
        lines.append(f"Note: {note}")
    lines.append(f"Предсказанное направление: {direction}")
    content = "\n".join(lines) + "\n"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    tmp_file.write_text(content, encoding="utf-8")
    tmp_file.replace(out_file)


def main() -> int:
    """Основной сценарий генерации прогноза на текущую дату.

    Загружает настройки и rules.yaml, проверяет существующий файл прогноза,
    читает сегодняшний sentiment из PKL, вычисляет направление и пишет итоговый
    txt. Любая штатная или нештатная ситуация заканчивается return 0, чтобы
    сбой sentiment-этапа не останавливал внешний пайплайн.
    """
    logger = setup_logging()

    try:
        # --- Загрузка настроек из <ticker>/settings.yaml (common + sentiment-секция) ---
        settings = load_settings(TICKER_DIR / "settings.yaml")

        predict_path = Path(settings["predict_path"])
        predict_path.mkdir(parents=True, exist_ok=True)

        today = date.today()
        date_str = today.strftime("%Y-%m-%d")
        out_file = predict_path / f"{date_str}.txt"

        if out_file.exists():
            if should_rewrite_existing_predict(out_file, today, settings["time_start"]):
                logger.info(f"Файл {out_file} создан до {settings['time_start']} (тестовый) — перезаписываем.")
            else:
                logger.info(f"Файл {out_file} уже существует — пропуск.")
                return 0

        rules = load_rules(TICKER_DIR / "rules.yaml")

        pkl_path = resolve_sentiment_pkl(settings)

        try:
            sentiment = get_today_sentiment(pkl_path, today)
        except FileNotFoundError as exc:
            logger.error(f"pkl_missing: {exc}")
            write_predict(out_file, date_str, "skip", "pkl_missing", note=str(exc))
            return 0
        except ValueError as exc:
            msg = str(exc)
            status = "pkl_duplicate" if "несколько строк" in msg else "error"
            logger.error(f"{status}: {msg}")
            write_predict(out_file, date_str, "skip", status, note=msg)
            return 0

        if sentiment is None:
            logger.info(f"В pkl нет записи за {today}.")
            write_predict(out_file, date_str, "skip", "no_pkl_row",
                          note=f"в sentiment_scores.pkl нет строки за {date_str}")
            return 0

        action = match_action(sentiment, rules)
        if action is None:
            logger.info(f"{today}: sentiment={sentiment:.2f} не попал ни в один диапазон rules.yaml.")
            write_predict(out_file, date_str, "skip", "no_rule_match",
                          sentiment=sentiment,
                          note="sentiment вне всех диапазонов rules.yaml")
            return 0

        direction = resolve_direction(sentiment, action)
        logger.info(f"{today}: sentiment={sentiment:.2f}, action={action}, direction={direction}")

        if direction == "skip":
            status = "error"
            note = f"не удалось определить направление для action={action!r}"
            write_predict(out_file, date_str, "skip", status,
                          sentiment=sentiment, action=action, note=note)
            return 0

        write_predict(out_file, date_str, direction, "ok",
                      sentiment=sentiment, action=action)
        logger.info(f"Записан файл предсказания: {out_file}")
        return 0

    except Exception as exc:
        logger.exception("Необработанная ошибка sentiment_to_predict")
        try:
            today = date.today()
            date_str = today.strftime("%Y-%m-%d")
            merged = load_settings(TICKER_DIR / "settings.yaml")
            predict_path = Path(
                merged.get("predict_path", str(TICKER_DIR / "predict"))
            )
            out_file = predict_path / f"{date_str}.txt"
            write_predict(out_file, date_str, "skip", "error",
                          note=f"{type(exc).__name__}: {exc}")
        except Exception as write_exc:
            logger.error(f"Не удалось записать файл предсказания с ошибкой: {write_exc}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
