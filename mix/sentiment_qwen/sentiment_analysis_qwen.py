"""Собирает qwen-based sentiment-оценки новостей из markdown-файлов через Ollama.

Скрипт читает настройки из `mix/settings.yaml` (секции `common` и
`sentiment_qwen`), строит жесткий промпт для Qwen, вызывает локальный
`/api/generate`, строго парсит ответ модели как одно число и сохраняет
результаты в PKL.

Дополнительно скрипт:
- пересчитывает новые и измененные markdown-файлы по `content_hash`;
- пересчитывает самую свежую дату, чтобы не держать неполные данные;
- добавляет рыночные признаки из дневной SQLite-базы котировок;
- сохраняет `raw_response` для последующего аудита качества ответов модели.
"""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import tiktoken
import typer
import yaml

TICKER_DIR = Path(__file__).resolve().parents[1]

app = typer.Typer(help="Собирает sentiment-оценки новостей через локальную модель Qwen в Ollama.")

# DEFAULT_PROMPT_TEMPLATE = (
#     "Оцени влияние новости на {ticker} по шкале от -10 до +10.\n"
#     "-10 = сильно негативно, 0 = нейтрально, +10 = сильно позитивно.\n\n"
#     "Текст новости:\n{news_text}\n\n"
#     "Ответ: только одно целое число от -10 до +10."
# )

DEFAULT_PROMPT_TEMPLATE = (
    "Оцени влияние на {ticker} от -10 до +10.\n\n"
    "Текст новости:\n\n{news_text}\n\n"
    "Верни только одно число от -10 до +10 без пояснений."
)

DEFAULT_TOKEN_LIMIT = 16000
STRICT_NUMBER_REGEX = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*$")


def cleanup_old_logs(log_dir: Path, max_files: int = 3) -> None:
    """Удаляет старые log-файлы, оставляя только несколько самых свежих."""
    log_files = sorted(log_dir.glob("sentiment_analysis_qwen_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old_file in log_files[max_files:]:
        try:
            old_file.unlink()
        except OSError as exc:
            print(f"Не удалось удалить старый лог {old_file}: {exc}")


def setup_logging(ticker_label: str, verbose: bool = False) -> None:
    """Настраивает файловое и консольное логирование для запуска скрипта."""
    level = logging.DEBUG if verbose else logging.INFO
    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"sentiment_analysis_qwen_{timestamp}.txt"
    cleanup_old_logs(log_dir, max_files=3)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("[%s] Запуск sentiment_analysis_qwen. Лог: %s", ticker_label, log_file)


def load_settings() -> dict:
    """Загружает и нормализует настройки из common и sentiment_qwen."""
    raw = yaml.safe_load((TICKER_DIR / "settings.yaml").read_text(encoding="utf-8"))
    settings = {**(raw.get("common") or {}), **(raw.get("sentiment_qwen") or {})}
    ticker = settings.get("ticker", "")
    ticker_lc = settings.get("ticker_lc", ticker.lower())
    for key, value in list(settings.items()):
        if isinstance(value, str):
            settings[key] = value.replace("{ticker}", ticker).replace("{ticker_lc}", ticker_lc)
    return settings


def find_md_files(md_dir: Path) -> list[Path]:
    """Возвращает отсортированный список markdown-файлов в каталоге новостей."""
    return sorted(path for path in md_dir.rglob("*.md") if path.is_file())


def read_markdown(path: Path) -> str:
    """Читает markdown-файл как UTF-8 текст и убирает лишние пробелы по краям."""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def compute_content_hash(path: Path) -> str:
    """Считает SHA-256 хэш содержимого файла для контроля изменений."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_prompt(ticker: str, prompt_template: str, news_text: str) -> str:
    """Подставляет тикер и текст новости в шаблон промпта."""
    return prompt_template.format(ticker=ticker, news_text=news_text)


def get_token_count(text: str) -> int:
    """Оценивает число токенов в тексте через tiktoken."""
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def warn_if_token_limit_exceeded(prompt: str, token_limit: int, file_name: str) -> int:
    """Логирует предупреждение, если промпт превышает заданный порог токенов."""
    prompt_tokens = get_token_count(prompt)
    if prompt_tokens > token_limit:
        logging.warning(
            "Prompt для %s содержит %s токенов, превышает порог %s. Возможны обрезание или плохой ответ.",
            file_name,
            prompt_tokens,
            token_limit,
        )
    return prompt_tokens


def round_half_away_from_zero(value: float) -> int:
    """Округляет число до ближайшего целого по правилу half away from zero."""
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def parse_sentiment_strict(response: str) -> Optional[int]:
    """Строго парсит ответ модели как одно число и возвращает целый sentiment."""
    if not response:
        return None
    match = STRICT_NUMBER_REGEX.fullmatch(response)
    if not match:
        return None
    value = match.group(1).replace(",", ".")
    try:
        score = float(value)
    except ValueError:
        return None
    rounded = round_half_away_from_zero(score)
    return max(min(rounded, 10), -10)


def extract_date_from_path(path: Path) -> Optional[str]:
    """Извлекает дату формата YYYY-MM-DD из пути к markdown-файлу."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(path))
    return match.group(1) if match else None


def run_ollama(model: str, prompt: str, keepalive: Optional[str] = None, timeout: int = 600) -> str:
    """Вызывает Ollama HTTP API с детерминированными параметрами генерации."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
        },
    }
    if keepalive:
        payload["keep_alive"] = keepalive
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return (response.json().get("response") or "").strip()


def load_existing_results(path: Path) -> pd.DataFrame:
    """Загружает ранее сохраненные результаты из PKL, если файл существует."""
    if not path.exists():
        return pd.DataFrame()
    with path.open("rb") as file_obj:
        return pd.DataFrame(pickle.load(file_obj))


def should_process_file(md_file: Path, existing_df: pd.DataFrame) -> bool:
    """Определяет, нужно ли пересчитывать файл по наличию и content_hash."""
    if existing_df.empty:
        return True
    md_file_path = str(md_file.resolve())
    matches = existing_df[existing_df["file_path"] == md_file_path]
    if matches.empty:
        return True
    current_hash = compute_content_hash(md_file)
    stored_hash = matches.iloc[-1].get("content_hash")
    return stored_hash != current_hash


def drop_latest_date(existing_df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет самую свежую дату из старых результатов для безопасного пересчета."""
    if existing_df.empty or "source_date" not in existing_df.columns:
        return existing_df
    max_date = existing_df["source_date"].max()
    if not max_date:
        return existing_df
    before = len(existing_df)
    trimmed = existing_df[existing_df["source_date"] != max_date].reset_index(drop=True)
    logging.info("Удалена последняя запись за %s (%s -> %s строк) для пересчета.", max_date, before, len(trimmed))
    return trimmed


def attach_market_features(df: pd.DataFrame, quotes_path: Path) -> pd.DataFrame:
    """Добавляет к sentiment-таблице рыночные признаки из SQLite-базы котировок."""
    if df.empty:
        return df
    if not quotes_path.exists():
        logging.warning("Файл котировок не найден: %s. Пропускаю добавление market features.", quotes_path)
        return df

    with sqlite3.connect(str(quotes_path)) as conn:
        quotes_df = pd.read_sql_query(
            "SELECT TRADEDATE, OPEN, CLOSE FROM Futures",
            conn,
            parse_dates=["TRADEDATE"],
        )

    quotes_df = quotes_df.dropna(subset=["TRADEDATE", "OPEN", "CLOSE"]).sort_values("TRADEDATE").reset_index(drop=True)
    quotes_df["body"] = quotes_df["CLOSE"] - quotes_df["OPEN"]
    quotes_df["date_only"] = quotes_df["TRADEDATE"].dt.date

    quote_dates = np.array(quotes_df["date_only"].tolist())
    bodies = quotes_df["body"].to_numpy()
    opens = quotes_df["OPEN"].to_numpy()

    def body_for(date_value):
        if date_value is None:
            return None
        idx = np.searchsorted(quote_dates, date_value)
        if idx < len(quote_dates) and quote_dates[idx] == date_value:
            return float(bodies[idx])
        return None

    def next_body_for(date_value):
        if date_value is None:
            return None
        idx = np.searchsorted(quote_dates, date_value, side="right")
        if idx < len(quote_dates):
            return float(bodies[idx])
        return None

    def next_open_to_open_for(date_value):
        if date_value is None:
            return None
        idx = np.searchsorted(quote_dates, date_value, side="right")
        if idx + 1 < len(opens):
            return float(opens[idx + 1] - opens[idx])
        return None

    def parse_date(value):
        if value is None:
            return None
        try:
            return datetime.strptime(str(value), "%Y-%m-%d").date()
        except ValueError:
            return None

    result_df = df.copy()
    result_df["date"] = result_df["source_date"].apply(parse_date)
    result_df["body"] = result_df["date"].apply(body_for)
    result_df["next_body"] = result_df["date"].apply(next_body_for)
    result_df["next_open_to_open"] = result_df["date"].apply(next_open_to_open_for)
    return result_df


def save_results(path: Path, df: pd.DataFrame) -> None:
    """Сохраняет итоговый датафрейм результатов в PKL-файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(df, file_obj)
    logging.info("Saved %s records to %s", len(df), path)


@app.command()
def main(
    output_pkl: Optional[Path] = typer.Option(
        None,
        help="Файл для сохранения sentiment-оценок. Если не задан, берется из settings.yaml.",
    ),
    model: Optional[str] = typer.Option(
        None,
        help="Локальная модель Ollama. По умолчанию берется из settings.yaml:sentiment_qwen.sentiment_model.",
    ),
    keepalive: str = typer.Option("5m", help="Удерживать модель Ollama загруженной между запросами."),
    token_limit: int = typer.Option(DEFAULT_TOKEN_LIMIT, help="Порог токенов для предупреждения о длинном prompt."),
    prompt_template: str = typer.Option(DEFAULT_PROMPT_TEMPLATE, help="Шаблон промпта для модели."),
    resume: bool = typer.Option(True, help="Пропускать уже обработанные и неизмененные файлы."),
    verbose: bool = typer.Option(False, help="Включить подробный лог."),
) -> None:
    """Запускает полный пайплайн расчета sentiment-оценок для Qwen."""
    settings = load_settings()
    ticker = settings.get("ticker", "RTS")
    setup_logging(ticker, verbose)

    if model is None:
        model = settings.get("sentiment_model", "qwen2.5:14b")
    logging.info("Sentiment model: %s", model)

    md_path = Path(settings.get("md_path", "."))
    sentiment_output = Path(settings.get("sentiment_output_pkl", "sentiment_scores.pkl"))
    if output_pkl is None:
        output_pkl = sentiment_output
    if not output_pkl.is_absolute():
        output_pkl = TICKER_DIR / output_pkl

    if not md_path.exists():
        raise typer.BadParameter(f"Папка markdown-файлов не найдена: {md_path}")

    files = find_md_files(md_path)
    if not files:
        typer.echo("В папке не найдено markdown-файлов.")
        raise typer.Exit(code=1)

    logging.info("Found %s markdown files in %s", len(files), md_path)

    existing_df = load_existing_results(output_pkl) if resume else pd.DataFrame()
    if resume:
        existing_df = drop_latest_date(existing_df)

    rows_by_path: dict[str, dict] = {}
    if not existing_df.empty:
        for row in existing_df.to_dict("records"):
            rows_by_path[row["file_path"]] = row

    for md_file in files:
        md_file_path = str(md_file.resolve())
        if resume and not should_process_file(md_file, existing_df):
            logging.info("[%s] Skipping unchanged file: %s", ticker, md_file.name)
            continue

        logging.info("[%s] Processing file: %s", ticker, md_file.name)
        news_text = read_markdown(md_file)
        prompt = build_prompt(ticker, prompt_template, news_text)
        prompt_tokens = warn_if_token_limit_exceeded(prompt, token_limit, md_file.name)
        content_hash = compute_content_hash(md_file)

        try:
            raw_response = run_ollama(model=model, prompt=prompt, keepalive=keepalive)
            sentiment = parse_sentiment_strict(raw_response)
        except Exception as exc:
            logging.error("Error processing %s: %s", md_file.name, exc)
            raw_response = str(exc)
            sentiment = None

        rows_by_path[md_file_path] = {
            "file_path": md_file_path,
            "content_hash": content_hash,
            "source_date": extract_date_from_path(md_file),
            "ticker": ticker,
            "model": model,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "raw_response": raw_response,
            "sentiment": sentiment,
            "processed_at": datetime.now(timezone.utc),
        }

        logging.info(
            "[%s] Result %s: sentiment=%s, prompt_tokens=%s",
            ticker,
            md_file.name,
            sentiment,
            prompt_tokens,
        )

    df = pd.DataFrame(rows_by_path.values())

    if not df.empty and "source_date" in df.columns:
        before = len(df)
        df = (
            df.sort_values(["source_date", "processed_at"], kind="stable")
            .drop_duplicates(subset="source_date", keep="last")
            .reset_index(drop=True)
        )
        if len(df) < before:
            logging.info("Дедуп по source_date: %s -> %s строк", before, len(df))

    path_db_day_str = settings.get("path_db_day", "")
    if path_db_day_str:
        df = attach_market_features(df, Path(path_db_day_str))

    save_results(output_pkl, df)
    typer.echo(f"Готово: {len(df)} записей сохранено в {output_pkl}")

    console_cols = [
        "source_date",
        "ticker",
        "model",
        "content_hash",
        "sentiment",
        "body",
        "next_body",
        "next_open_to_open",
        "prompt_tokens",
    ]
    console_df = df[[col for col in console_cols if col in df.columns]]
    typer.echo("\nРезультаты:")
    typer.echo(console_df.to_string(index=False))


if __name__ == "__main__":
    app()
