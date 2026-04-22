"""
Чтение текущих фьючерсных позиций из двух источников:

1. trade/state/positions.yaml — ручной override (приоритетный).
   Если в нём есть раскомментированные данные для запрошенного счёта/тикера,
   возвращается значение оттуда.

2. trade/quik_export/positions.json — автоматический экспорт из QUIK
   (quik_export_positions.lua). Используется, если в YAML данных нет.

Если позиция не найдена ни в одном источнике — возвращается 0 (вне рынка).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_TRADE_DIR = Path(__file__).resolve().parent
_YAML_PATH = _TRADE_DIR / "state" / "positions.yaml"
_JSON_PATH = _TRADE_DIR / "quik_export" / "positions.json"


def _read_yaml(trdaccid: str, sec_code: str) -> int | None:
    """Пытается прочитать totalnet из positions.yaml. None — если данных нет."""
    if not _YAML_PATH.exists():
        return None
    try:
        data = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"positions.yaml: ошибка чтения: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    account = data.get(trdaccid)
    if not isinstance(account, dict):
        return None
    ticker = account.get(sec_code)
    if not isinstance(ticker, dict):
        return None
    val = ticker.get("totalnet")
    if val is None:
        return None
    return int(val)


def _read_json(trdaccid: str, sec_code: str) -> int | None:
    """Пытается прочитать totalnet из positions.json (LUA-экспорт)."""
    if not _JSON_PATH.exists():
        return None
    try:
        data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"positions.json: ошибка чтения: {exc}")
        return None
    for pos in data.get("positions", []):
        if pos.get("trdaccid") == trdaccid and pos.get("sec_code") == sec_code:
            return int(pos["totalnet"])
    return None


def get_position(trdaccid: str, sec_code: str) -> int:
    """
    Возвращает чистую позицию (totalnet) по торговому счёту и тикеру.

    Приоритет: positions.yaml → positions.json → 0.
    Положительное значение = лонг, отрицательное = шорт, 0 = вне рынка.
    """
    # 1. Ручной override
    val = _read_yaml(trdaccid, sec_code)
    if val is not None:
        logger.info(f"Позиция {sec_code}@{trdaccid} из YAML (override): {val}")
        return val

    # 2. LUA-экспорт
    val = _read_json(trdaccid, sec_code)
    if val is not None:
        logger.info(f"Позиция {sec_code}@{trdaccid} из JSON (QUIK): {val}")
        return val

    # 3. Нет данных — считаем, что вне рынка
    logger.info(f"Позиция {sec_code}@{trdaccid}: нет данных, принимаем 0 (вне рынка)")
    return 0


def has_yaml_override(trdaccid: str, sec_code: str) -> bool:
    """True, если в positions.yaml есть раскомментированный totalnet для (trdaccid, sec_code)."""
    return _read_yaml(trdaccid, sec_code) is not None


def is_export_fresh(today: date) -> bool:
    """True, если positions.json экспортирован сегодня (по дате из exported_at).

    Нужно для защиты от торговли по устаревшим позициям, если LUA-экспортёр
    не отработал (QUIK выключен, окно терминала закрыто и т.п.).
    Отсутствие файла или невалидный формат → False.
    """
    if not _JSON_PATH.exists():
        return False
    try:
        data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
        exported_at = data.get("exported_at")
        if not exported_at:
            return False
        export_date = datetime.strptime(exported_at, "%Y-%m-%d %H:%M:%S").date()
        return export_date == today
    except Exception as exc:
        logger.warning(f"positions.json: не удалось распарсить exported_at: {exc}")
        return False


def get_exported_at() -> str | None:
    """Возвращает метку времени последнего LUA-экспорта или None."""
    if not _JSON_PATH.exists():
        return None
    try:
        data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
        return data.get("exported_at")
    except Exception:
        return None
