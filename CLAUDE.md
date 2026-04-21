# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`pj18_sentiment` — исследовательский пайплайн по влиянию новостного фона на настроение рынка. Для каждого торгового инструмента (сейчас `rts` и `mix`) прогоняются две локальные LLM (gemma3 и qwen) через Ollama, результаты превращаются в сигналы и бэктестятся против дневных котировок фьючерсов.

Внешние зависимости (не в репо):
- Ollama с моделями `gemma3:12b`, `qwen2.5:14b`, `qwen3:14b` — HTTP API на `localhost`.
- SQLite-базы котировок и новостей в `C:/Users/Alkor/gd/...` (см. `*/settings.yaml`).
- Markdown-файлы новостей в `md_path` из `settings.yaml`.

## Commands

Виртуальное окружение: `.venv/` (Windows). Активация: `.venv\Scripts\activate` (cmd) или `. .venv/Scripts/activate` (bash).

- Полный отчётный прогон: `python run_report.py` — порядок шагов захардкожен в `HARD_STEPS` / `SOFT_STEPS` в [run_report.py](run_report.py); лог пишется в `log/run_report_<timestamp>.txt`, хранятся последние 3.
- Открыть все HTML-отчёты бэктестов в Chrome: `python html_open.py`.
- Отдельный этап: запуск любого `sentiment_*.py` напрямую, например `python rts/sentiment_qwen/sentiment_analysis_qwen.py` — все скрипты этапов используют `typer`, поддерживают `--help`.
- Тесты (pytest, запускаются из корня, чтобы сработали импорты вида `from rts.sentiment_qwen import ...`):
  - весь пакет: `pytest`
  - один файл: `pytest rts/sentiment_qwen/test_sentiment_analysis_qwen.py`
  - один тест: `pytest rts/sentiment_qwen/test_sentiment_analysis_qwen.py::test_parse_sentiment_strict_accepts_only_single_numeric_value`

## Architecture

### Параллельные ветки `<ticker>/<model>/`

`rts/` и `mix/` — независимые тикеры; внутри каждого — по две sentiment-ветки (`sentiment_gemma`, `sentiment_qwen`) с одинаковой тройкой этапов. Новый тикер или модель добавляется копированием этого шаблона + регистрацией в `HARD_STEPS`/`SOFT_STEPS` в `run_report.py`. Код веток не делит общий модуль — они преднамеренно дублируются, правки часто нужно применять зеркально во всех четырёх каталогах.

### Этапы пайплайна (выполняются в этом порядке для каждой ветки)

1. `sentiment_analysis_*.py` — читает markdown-файлы новостей, вызывает Ollama `/api/generate`, строго парсит один числовой ответ в диапазоне [-10; 10], считает `content_hash` по содержимому md и сохраняет PKL (`sentiment_scores_<model>.pkl`). Повторные прогоны пропускают уже обработанные файлы по хэшу (`use_cache: true`). Обогащает PKL рыночными признаками из дневной SQLite.
2. `sentiment_group_stats.py` — агрегирует скоры по группам и сохраняет статистики/графики в `group_stats/` и `plots/`.
3. `sentiment_backtest.py` — применяет `rules.yaml` ветки (см. ниже), строит equity и QuantStats-отчёт: `plots/sentiment_backtest.html` и `plots/sentiment_backtest_qs.html`.
4. Soft-этап `<ticker>/combine/sentiment_compare.py` — объединяет результаты gemma и qwen в сравнительный отчёт `plots/sentiment_compare.html`.

### Конфигурация

`<ticker>/settings.yaml` — единственная точка настройки ветки. Структура: секция `common` + по секции на каждую модельную ветку (`sentiment_gemma`, `sentiment_qwen`, `combined`). Специфичные секции накрывают одноимённые ключи `common`. Плейсхолдеры `{ticker}` / `{ticker_lc}` форматируются в путях.

`<ticker>/<model>/rules.yaml` — правила перевода sentiment-числа в торговое действие (`follow`/`invert`/`skip`) для бэктеста. Правила матчатся сверху вниз по первому попаданию; непокрытые диапазоны = skip. Файл задаётся отдельно под каждую конкретную модель (см. закомментированные пресеты под `qwen2.5:14b` vs `qwen3:14b`).

### Оркестрация и логи

`run_report.py` + [orchestrator_logging.py](orchestrator_logging.py): запускают дочерние Python-процессы через `subprocess`, HARD-шаг при ненулевом коде/исключении валит весь прогон (`sys.exit`), SOFT-шаг только логируется. Логирование двухканальное: файл — без ANSI, консоль — с цветами по уровню (ERROR=красный, WARNING=жёлтый, `OK` в сообщении=зелёный). Используй `build_handlers(log_file)` при добавлении новых оркестраторов.

Каждый этапный скрипт сам ведёт лог в `<branch>/log/` и ротирует до 3 последних файлов через `cleanup_old_logs`.

### `beget/`

Отдельный под-проект (не часть sentiment-пайплайна): скраперы RSS на удалённом Linux-сервере (`beget/server/*.py`) и `sync_files.py` для выкачивания `.db`/`.log` на Windows через `wsl rsync`. Не запускается `run_report.py`.
