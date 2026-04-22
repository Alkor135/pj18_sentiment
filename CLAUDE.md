# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`pj18_sentiment` — автоматизированная торговля фьючерсами MOEX на основе sentiment-анализа новостного фона. Для каждого инструмента (сейчас `rts` и `mix`) прогоняются две локальные LLM (gemma3 и qwen) через Ollama, по каждой модели формируется прогноз направления на текущую торговую дату, результат превращается в торговый ордер QUIK через `.tri`-файл. Параллельно считаются бэктесты и аналитические отчёты.

Внешние зависимости (не в репо):

- Ollama с моделями `gemma3:12b`, `qwen3:14b` (альтернативно `qwen2.5:14b`) — HTTP API на `localhost:11434`.
- SQLite-базы котировок и новостей в `C:/Users/Alkor/gd/...` (см. `<ticker>/settings.yaml`).
- Markdown-файлы новостей в `md_path` из `settings.yaml` (общий каталог для всех тикеров).
- QUIK с `algotrade/input.tri` и lua-скриптами [quik_export_minutes.lua](trade/quik_export_minutes.lua), [quik_export_positions.lua](trade/quik_export_positions.lua).
- Удалённый Linux-сервер со скраперами RSS — см. [beget/](beget/).

## Commands

Виртуальное окружение: `.venv/` (Windows). Активация: `.venv\Scripts\activate` (cmd) или `. .venv/Scripts/activate` (bash).

- **Боевой/тестовый запуск полного пайплайна**: `python run_all.py` — единственная точка входа, регистрируется в Windows Task Scheduler на 21:00:05 ежедневно. Порядок шагов — в `HARD_STEPS` / `SOFT_STEPS` в [run_all.py](run_all.py); лог `log/run_all_<timestamp>.txt`, хранятся последние 3.
- **Переигровка только отчётной части** (без торговли и загрузки данных): `python run_report.py` — этапы анализа настроений + бэктест + compare; шаги захардкожены в [run_report.py](run_report.py).
- **Открыть все HTML-отчёты бэктестов в Chrome**: `python html_open.py`.
- **Отдельный этап**: запуск любого `sentiment_*.py` / `shared/*.py` / `trade/trade_*.py` напрямую — скрипты этапов используют `typer` (где применимо) и поддерживают `--help`.
- **Ручные операции** (ролловер фьючерса, смена количества контрактов) описаны в [OPERATIONS.md](OPERATIONS.md).
- **Тесты** (pytest, запускаются из корня, чтобы сработали импорты вида `from rts.sentiment_qwen import ...`):
  - весь пакет: `pytest`
  - один файл: `pytest rts/sentiment_qwen/test_sentiment_analysis_qwen.py`
  - один тест: `pytest rts/sentiment_qwen/test_sentiment_analysis_qwen.py::test_parse_sentiment_strict_accepts_only_single_numeric_value`

## Architecture

### Параллельные ветки `<ticker>/<model>/`

`rts/` и `mix/` — независимые тикеры; внутри каждого — по две sentiment-ветки (`sentiment_gemma`, `sentiment_qwen`) с одинаковым набором этапов плюс общая ветка `combine/` для сравнительного отчёта. Новый тикер или модель добавляется копированием этого шаблона + регистрацией в `HARD_STEPS`/`SOFT_STEPS` в [run_all.py](run_all.py) и [run_report.py](run_report.py). Код веток не делит общий модуль — они преднамеренно дублируются, правки часто нужно применять зеркально во всех четырёх каталогах.

### Этапы пайплайна `run_all.py`

Шаги HARD (ненулевой код → `sys.exit`):

0. [prepare.py](prepare.py) — если запуск **до 21:00**, удаляет сегодняшние результаты (прогнозы `<ticker>_sentiment_<model>/YYYY-MM-DD.txt` и все сегодняшние done-маркеры в `trade/state/`), чтобы дневное тестирование прогонов было идемпотентным. После 21:00 трогает только test-done маркеры. Housekeeping: `trade/state/*.done` ≤ `DONE_RETENTION_DAYS` (10) и ≤ `DONE_RETENTION_FILES` (10); `log/prepare_*.txt` — последние 3.
1. [beget/sync_files.py](beget/sync_files.py) — синхронизация `.db` и `.log` новостей + `.tri` из QUIK с удалённого сервера через `wsl rsync`. Конфиг — [beget/settings.yaml](beget/settings.yaml).
2. `<ticker>/shared/download_minutes_to_db.py` — подкачка минутных котировок в SQLite `path_db_minute` (источник — CSV `quik_csv_path` от [trade/quik_export_minutes.lua](trade/quik_export_minutes.lua)).
3. `<ticker>/shared/convert_minutes_to_days.py` — агрегация минут → дневные свечи (`time_start`..`time_end` = сессия 21:00 предыдущего дня..20:59:59 текущего) в `path_db_day`.
4. [rts/shared/create_markdown_files.py](rts/shared/create_markdown_files.py) — markdown-сводки новостей по торговым сессиям из БД провайдера, фильтр по ключам «нефт»/«газ». Пишет в `md_path` (общий для всех тикеров — вызов из `mix/shared/` избыточен и не делается).
5. `<ticker>/sentiment_<model>/sentiment_analysis_<model>.py` (для каждой модели — gemma и qwen) — жёсткий промпт к Ollama `/api/generate`, строго парсит один числовой ответ в диапазоне [-10; 10], считает `content_hash` по содержимому md и сохраняет PKL (`sentiment_scores.pkl`). Повторные прогоны пропускают уже обработанные файлы по хэшу (`use_cache: true`). Обогащает PKL рыночными признаками из дневной SQLite.
6. `<ticker>/sentiment_<model>/sentiment_to_predict.py` — берёт строку sentiment за сегодня, применяет `rules.yaml` (см. ниже) и пишет `<predict_path>/YYYY-MM-DD.txt`. Файл пишется **всегда**, даже при нештатной ситуации (направление = skip, причина в поле Status: `ok` / `no_rule_match` / `no_pkl_row` / `pkl_missing` / `pkl_duplicate` / `error`). Если файл уже есть и создан после `time_start` — не перезаписывается. Возвращает 0 даже при сбое, чтобы один тикер не валил пайплайн.
7. `trade/trade_<ticker>_sentiment_<model>_<account>.py` — target-state исполнение сделок через `.tri`-файлы QUIK. См. ниже.

Шаги SOFT (ошибка → warning, продолжаем):

- `<ticker>/sentiment_<model>/sentiment_group_stats.py` — агрегаты скоров по группам, `group_stats/` и `plots/`.
- `<ticker>/sentiment_<model>/sentiment_backtest.py` — применяет `rules.yaml`, строит equity и QuantStats-отчёт: `plots/sentiment_backtest.html`, `plots/sentiment_backtest_qs.html`, XLSX в `backtest/`.
- `<ticker>/combine/sentiment_compare.py` — объединяет результаты gemma и qwen по `source_date`, equity-кривые обеих стратегий и их простой комбинации → `plots/sentiment_compare.html`.

### Торговая подсистема `trade/`

- [trade/settings.yaml](trade/settings.yaml) — аккаунты QUIK (`iis` / `ebs`) с их `trade_path`, `trade_account` и `quantity_close` / `quantity_open` по каждому тикеру.
- `trade_<ticker>_sentiment_<model>_<account>.py` — target-state логика: читает прогноз на сегодня → вычисляет целевую позицию (up → +qty, down → −qty, skip → текущая сохраняется) → из `read_positions.py` получает текущую позицию → пишет дельту (закрытие + открытие) в `trade_path/input.tri`. При ролловере (`ticker_close ≠ ticker_open`) закрывает старый контракт и открывает новый. Особенность gemma-скрипта: отсутствие файла прогноза трактуется как «молчим», позиция сохраняется.
- [trade/read_positions.py](trade/read_positions.py) — текущая позиция в двух источниках: приоритетный ручной override в `trade/state/positions.yaml`, иначе автоэкспорт `trade/quik_export/positions.json` от [quik_export_positions.lua](trade/quik_export_positions.lua); нет данных → 0 (вне рынка). Проверяется свежесть экспорта.
- Защита от двойной записи: маркеры `trade/state/<ticker>_<trade_account>_sentiment_<model>[_test]_<YYYY-MM-DD>.done`. Переключение между боевым и тестовым режимом — через раскомментирование `trade_filepath = trade_path / "test.tri"` в начале торгового скрипта.
- [trade/quik_export/](trade/quik_export/) (`minutes.csv`, `positions.json`) игнорируется git; соответствующие `.lua` лежат в `trade/`.

### Конфигурация

`<ticker>/settings.yaml` — единственная точка настройки ветки. Структура: секция `common` + по секции на каждую модельную ветку (`sentiment_gemma`, `sentiment_qwen`, `combined`). Специфичные секции накрывают одноимённые ключи `common`. Плейсхолдеры `{ticker}` / `{ticker_lc}` форматируются в путях. Ключи ролловера — `ticker_close` / `ticker_open` (см. [OPERATIONS.md](OPERATIONS.md)).

`<ticker>/<model>/rules.yaml` — правила перевода sentiment-числа в торговое действие (`follow`/`invert`/`skip`) для `sentiment_to_predict.py` и `sentiment_backtest.py`. Правила матчатся сверху вниз по первому попаданию; непокрытые диапазоны = skip. Под каждую модель отдельный файл (см. закомментированные пресеты под `qwen2.5:14b` / `qwen3:14b` / `qwen3:14b RTX 3090Ti`).

[trade/settings.yaml](trade/settings.yaml) — счета и объёмы, общие для всех торговых скриптов.

### Оркестрация и логи

[run_all.py](run_all.py) / [run_report.py](run_report.py) + [orchestrator_logging.py](orchestrator_logging.py): запускают дочерние Python-процессы через `subprocess`, HARD-шаг при ненулевом коде/исключении валит весь прогон (`sys.exit`), SOFT-шаг только логируется. Логирование двухканальное: файл — без ANSI, консоль — с цветами по уровню (ERROR=красный, WARNING=жёлтый, `OK` в сообщении=зелёный). Используй `build_handlers(log_file)` при добавлении новых оркестраторов.

Каждый этапный скрипт сам ведёт лог в `<branch>/log/` и ротирует до 3 последних файлов через `cleanup_old_logs`.

### `beget/`

Отдельный под-проект: скраперы RSS на удалённом Linux-сервере (`beget/server/*.py`) и [beget/sync_files.py](beget/sync_files.py) для выкачивания `.db`/`.log` на Windows через `wsl rsync`. `sync_files.py` — часть основного пайплайна (шаг 1 в `run_all.py`); серверные скрипты деплоятся на сервер и из `run_all.py` не вызываются.
