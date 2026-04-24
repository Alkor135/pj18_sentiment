"""
Утилита для очистки папок __pycache__ в корне проекта.

Рекурсивно обходит дерево от каталога скрипта, удаляет каждую найденную
папку __pycache__ через shutil.rmtree и печатает её относительный путь.
Папки внутри SKIP_DIRS (по умолчанию — .venv и .git) не трогаются,
их пути выводятся с префиксом "skip:".

Запуск: `python clean_pycache.py` из корня проекта.
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent


SKIP_DIRS = {".venv", ".git"}

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

if sys.platform == "win32":
    os.system("")


def main() -> None:
    """Обходит проект, удаляет __pycache__ вне SKIP_DIRS, печатает отчёт."""
    removed = 0
    skipped = 0
    for path in ROOT.rglob("__pycache__"):
        if not path.is_dir():
            continue
        rel = path.relative_to(ROOT)
        if SKIP_DIRS & set(rel.parts):
            print(f"{GREEN}skip: {rel}{RESET}")
            skipped += 1
            continue
        shutil.rmtree(path)
        print(f"{RED}{rel}{RESET}")
        removed += 1
    print(f"Removed: {removed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
