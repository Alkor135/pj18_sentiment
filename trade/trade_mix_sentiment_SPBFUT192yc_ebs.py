"""
Исполнение сделок по фьючерсу MIX в QUIK через .tri-файлы — sentiment-стратегия.

Target-state модель:
  1. Читает sentiment-прогноз текущего дня.
  2. По сигналу вычисляет целевую позицию (up → +qty, down → -qty).
  3. Из read_positions.py получает текущую позицию (тикер + количество).
  4. Дельта = цель − текущая → пишет закрытие (противоположный ордер) + открытие (нужный ордер).
  5. При ролловере (ticker_close ≠ ticker_open): закрывает старый, открывает новый.

Отличия от combo:
  - читает секцию `sentiment` (predict_path → <ticker>_sentiment/);
  - sentiment_to_predict.py НЕ создаёт файл на skip → отсутствие файла трактуется
    как «молчим»: выходим без изменений, текущая позиция сохраняется (поведение
    по договорённости с пользователем, отличается от combo).

Поддержка ручного override позиций через trade/state/positions.yaml.
Логирование с ротацией (3 файла). Защита от двойной записи через маркер
state/{ticker}_{trade_account}_sentiment_{date}.done.
"""

from pathlib import Path
from datetime import datetime, date
import re
import logging
import sys
import yaml

# --- Импорт read_positions ---
_TRADE_DIR = Path(__file__).resolve().parent
if str(_TRADE_DIR) not in sys.path:
    sys.path.insert(0, str(_TRADE_DIR))
from read_positions import get_position, get_exported_at, is_export_fresh

# --- Конфигурация из mix/settings.yaml (common + sentiment) ---
ticker_lc = 'mix'
TICKER_DIR = Path(__file__).resolve().parents[1] / ticker_lc
_raw = yaml.safe_load((TICKER_DIR / "settings.yaml").read_text(encoding="utf-8"))
cfg = {**(_raw.get("common") or {}), **(_raw.get("sentiment") or {})}
_t = cfg.get("ticker", "")
_tl = cfg.get("ticker_lc", _t.lower())
for _k, _v in list(cfg.items()):
    if isinstance(_v, str):
        cfg[_k] = _v.replace("{ticker}", _t).replace("{ticker_lc}", _tl)

trade_settings_path = Path(__file__).parent / 'settings.yaml'
with open(trade_settings_path, encoding='utf-8') as f:
    trade_cfg = yaml.safe_load(f)

ticker_close = cfg['ticker_close']
ticker_open = cfg['ticker_open']

account = trade_cfg['accounts']['ebs']
trade_account = account['trade_account']
quantity_open = int(account[ticker_lc].get('quantity_open', 1))

# Пути к файлам
predict_path = Path(cfg['predict_path'])
log_path = Path(__file__).parent / "log"
trade_path = Path(account['trade_path'])
trade_filepath = trade_path / "input.tri"
# trade_filepath = trade_path / "test.tri"  # Для тестирования без реального QUIK (пишет в test.tri вместо input.tri)

# Создание необходимых директорий
trade_path.mkdir(parents=True, exist_ok=True)
log_path.mkdir(parents=True, exist_ok=True)
state_path = Path(__file__).parent / "state"
state_path.mkdir(parents=True, exist_ok=True)

# Имя файла прогноза на текущую дату
today = date.today()
current_filename = today.strftime("%Y-%m-%d") + ".txt"
current_filepath = predict_path / current_filename

# --- Настройка логгирования ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_path / f'trade_{ticker_lc}_sentiment_{timestamp}.txt'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Очистка старых логов (оставляем только 3 самых новых)
def cleanup_old_logs(log_dir: Path, prefix: str, max_files: int = 3):
    """Удаляет старые лог-файлы, оставляя max_files самых новых."""
    log_files = sorted(log_dir.glob(f"{prefix}_*.txt"))
    if len(log_files) > max_files:
        for old_file in log_files[:-max_files]:
            try:
                old_file.unlink()
                logger.info(f"Удалён старый лог: {old_file.name}")
            except Exception as e:
                logger.warning(f"Не удалось удалить {old_file}: {e}")

cleanup_old_logs(log_path, prefix=f"trade_{ticker_lc}_sentiment")

# --- Вспомогательные функции ---
def get_direction(filepath):
    """
    Извлекает предсказание (up/down/skip) из указанного файла.
    Проверяет несколько кодировок для корректного чтения.
    """
    encodings = ['utf-8', 'cp1251']
    for encoding in encodings:
        try:
            with filepath.open('r', encoding=encoding) as f:
                for line in f:
                    if "Предсказанное направление:" in line:
                        direction = line.split(":", 1)[1].strip().lower()
                        if direction in ['up', 'down', 'skip']:
                            return direction
            return None
        except UnicodeDecodeError:
            continue
    logger.error(f"Не удалось прочитать файл {filepath} с кодировками {encodings}.")
    return None

def get_next_trans_id(trade_filepath):
    """
    Определяет следующий TRANS_ID на основе максимального значения в файле.
    """
    trans_id = 1
    if trade_filepath.exists():
        try:
            with trade_filepath.open('r', encoding='cp1251') as f:
                content = f.read()
                trans_ids = re.findall(r'TRANS_ID=(\d+);', content)
                if trans_ids:
                    trans_id = max(int(tid) for tid in trans_ids if tid.isdigit()) + 1
        except (UnicodeDecodeError, ValueError) as e:
            logger.error(f"Ошибка при чтении TRANS_ID из {trade_filepath}: {e}")
    return trans_id

def create_trade_block(tr_id, ticker, action, quantity):
    """Формирует блок транзакции QUIK .tri-файла."""
    expiry_date = today.strftime("%Y%m%d")
    return (
        f'TRANS_ID={tr_id};'
        f'CLASSCODE=SPBFUT;'
        f'ACTION=Ввод заявки;'
        f'Торговый счет={trade_account};'
        f'К/П={action};'
        f'Тип=Рыночная;'
        f'Класс=SPBFUT;'
        f'Инструмент={ticker};'
        f'Цена=0;'
        f'Количество={quantity};'
        f'Условие исполнения=Поставить в очередь;'
        f'Комментарий={tr_id} {today.strftime("%y%m%d")};'
        f'Переносить заявку=Нет;'
        f'Дата экспирации={expiry_date};'
        f'Код внешнего пользователя=;\n'
    )

# --- Основная логика ---
# Защита от повторной записи: один тикер + одна дата + стратегия = один маркер
done_marker = state_path / f"{ticker_lc}_{trade_account}_sentiment_{today.strftime('%Y-%m-%d')}.done"
if done_marker.exists():
    logger.info(f"Маркер {done_marker.name} уже существует — транзакция за сегодня уже записана. Пропуск.\n")
    sys.exit(0)

# Проверка наличия файла прогноза на сегодня.
# Для sentiment отсутствие файла = «молчим»: оставляем текущую позицию без изменений
# (в отличие от combo, где skip → target=0). Файла нет ⇒ sentiment_to_predict.py
# намеренно его не создал (см. CLAUDE.md §3).
if not current_filepath.exists() or current_filepath.stat().st_size == 0:
    logger.info(f"Файл {current_filepath} не существует или пуст. Sentiment молчит — позиция сохраняется, ордеры не формируются.\n")
    sys.exit(0)

# Получение направления из текущего файла
current_predict = get_direction(current_filepath)

if current_predict is None:
    logger.warning("Не удалось найти предсказанное направление в файле.\n")
    sys.exit(0)

# Доп. защита: если внутри файла оказался skip — тоже «молчим», позицию не трогаем.
if current_predict == 'skip':
    logger.info("В файле прогноза skip — позиция сохраняется, ордеры не формируются.\n")
    sys.exit(0)

logger.info(f"Текущее предсказание: {current_predict} (файл: {current_filepath})")
logger.info(f"Источник позиций: LUA-экспорт из QUIK (если доступен), иначе positions.yaml")
exported_at = get_exported_at()
if exported_at:
    logger.info(f"LUA-экспорт: {exported_at}")

# Защита от устаревшего positions.json: если override через positions.yaml нет,
# а LUA-экспорт не обновлялся сегодня — останавливаем пайплайн (hard-fail).
# Причина: торговать по вчерашним позициям опаснее, чем пропустить день.
_yaml_override = Path(__file__).parent / "state" / "positions.yaml"
if not _yaml_override.exists() and not is_export_fresh(today):
    logger.error(
        f"positions.json не обновлялся сегодня ({today}). "
        f"Последний экспорт: {exported_at or 'n/a'}. "
        f"Проверь QUIK и quik_export_positions.lua. Остановка пайплайна."
    )
    sys.exit(1)

# --- Определение целевой позиции ---
# skip уже отсечён выше — остаются только up/down.
if current_predict == 'up':
    target_position = quantity_open
else:  # down
    target_position = -quantity_open

logger.info(f"Целевая позиция: {target_position} контрактов")

# --- Получение текущей позиции ---
# Для основного контракта (ticker_open)
current_position = get_position(trade_account, ticker_open)
logger.info(f"Текущая позиция {ticker_open}: {current_position} контрактов")

# --- Вычисление дельты и формирование заявок ---
delta = target_position - current_position
logger.info(f"Дельта (цель - текущая): {delta}")

if delta == 0:
    logger.info("Позиция уже в целевом состоянии. Ордеры не требуются.\n")
    done_marker.touch()
    sys.exit(0)

# Получаем TRANS_ID для первой заявки
trans_id = get_next_trans_id(trade_filepath)
trade_content = ""

# --- Логика: сначала закрываем, потом открываем ---
if delta > 0:
    # Нужно либо увеличить лонг, либо закрыть шорт
    if current_position < 0:
        # Сейчас в шорте — закрываем шорт противоположным ордером
        close_qty = abs(current_position)
        trade_content += create_trade_block(trans_id, ticker_open, 'Покупка', str(close_qty))
        trans_id += 1
        logger.info(f"  Закрытие шорта: Покупка {close_qty} контрактов {ticker_open}")

    # Если цель > 0, открываем/добавляем лонг
    if target_position > 0:
        open_qty = target_position - max(0, current_position)
        if open_qty > 0:
            trade_content += create_trade_block(trans_id, ticker_open, 'Покупка', str(open_qty))
            logger.info(f"  Открытие лонга: Покупка {open_qty} контрактов {ticker_open}")

elif delta < 0:
    # Нужно либо увеличить шорт, либо закрыть лонг
    if current_position > 0:
        # Сейчас в лонге — закрываем лонг противоположным ордером
        close_qty = current_position
        trade_content += create_trade_block(trans_id, ticker_open, 'Продажа', str(close_qty))
        trans_id += 1
        logger.info(f"  Закрытие лонга: Продажа {close_qty} контрактов {ticker_open}")

    # Если цель < 0, открываем/добавляем шорт
    if target_position < 0:
        open_qty = abs(target_position) - max(0, -current_position)
        if open_qty > 0:
            trade_content += create_trade_block(trans_id, ticker_open, 'Продажа', str(open_qty))
            logger.info(f"  Открытие шорта: Продажа {open_qty} контрактов {ticker_open}")

# --- Ролловер: если ticker_close ≠ ticker_open, закрываем позицию в старом контракте ---
if ticker_close != ticker_open:
    old_position = get_position(trade_account, ticker_close)
    if old_position != 0:
        trans_id += 1
        close_qty = abs(old_position)
        if old_position > 0:
            action = 'Продажа'
        else:
            action = 'Покупка'
        trade_content += create_trade_block(trans_id, ticker_close, action, str(close_qty))
        logger.info(f"  Ролловер: закрытие позиции {old_position} контрактов {ticker_close} ({action})")

# --- Запись результата ---
if trade_content:
    with trade_filepath.open('a', encoding='cp1251') as f:
        f.write(trade_content)
    done_marker.touch()
    logger.info(f"\nДобавлены заявки в файл {trade_filepath}.")
    logger.info(f"Сигнал: {current_predict}, переход {current_position} → {target_position}\n")
else:
    logger.info(f"На {today} никакие ордеры не требуются. Позиция уже совпадает с целью.\n")
    done_marker.touch()
