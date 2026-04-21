"""
Вспомогательный модуль для оркестраторов `run_all.py`, `run_report.py`
и `run_other.py`.

Назначение:
- формировать два независимых обработчика логов;
- оставлять файловый лог чистым, без ANSI-последовательностей;
- добавлять цвет только в консольный вывод, если терминал это поддерживает.

Цвета используются по простому правилу:
- `OK` в информационных сообщениях — зелёный;
- `warning` — жёлтый;
- `error` и более высокий уровень — красный.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def _stream_supports_color(stream) -> bool:
    """Проверяет, можно ли безопасно использовать ANSI-цвета в данном потоке."""
    is_a_tty = getattr(stream, "isatty", None)
    if not callable(is_a_tty) or not is_a_tty():
        return False

    if os.name != "nt":
        return True

    return bool(
        os.environ.get("WT_SESSION")
        or os.environ.get("ANSICON")
        or os.environ.get("ConEmuANSI") == "ON"
        or os.environ.get("TERM_PROGRAM")
        or os.environ.get("TERM")
    )


class PlainFormatter(logging.Formatter):
    """Обычный форматтер для файлового лога без цветовых управляющих кодов."""

    pass


class ColorConsoleFormatter(logging.Formatter):
    """Форматтер для консоли, который добавляет цвет по уровню и содержимому сообщения."""

    def __init__(self, fmt: str, use_color: bool = True):
        """Сохраняет формат сообщения и флаг, разрешающий цветной вывод."""
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует сообщение и при необходимости оборачивает его в ANSI-цвет."""
        message = super().format(record)
        if not self.use_color:
            return message

        color = ""
        if record.levelno >= logging.ERROR:
            color = RED
        elif record.levelno >= logging.WARNING:
            color = YELLOW
        elif "OK" in record.getMessage():
            color = GREEN

        if not color:
            return message

        return f"{color}{message}{RESET}"


def build_handlers(log_file: Path) -> list[logging.Handler]:
    """Создаёт обработчики для файла и консоли с разными форматтерами."""
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(PlainFormatter(LOG_FORMAT))

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(
        ColorConsoleFormatter(
            LOG_FORMAT,
            use_color=_stream_supports_color(stream_handler.stream),
        )
    )

    return [file_handler, stream_handler]
