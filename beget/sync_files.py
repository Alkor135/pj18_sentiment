"""
Надёжная синхронизация SQLite + Google Drive + rsync + WSL
Без code 23 и без Permission denied.

Особенности:
- SQLite-safe параметры rsync
- Google Drive safe (--inplace)
- Цветной вывод
- Лог файл
- stderr логируется
"""

import subprocess
from pathlib import Path
from datetime import datetime
import sys
import yaml

# Цвета консоли
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def get_timestamp():
    """Возвращает текущее время в формате для консольных сообщений и логов."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


with open(Path(__file__).parent / "settings.yaml", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

sync_configs = _cfg["sources"]
remote_host = _cfg["remote_host"]


def ensure_dir(directory: Path):
    """Создаёт директорию назначения вместе с родительскими папками при необходимости."""
    directory.mkdir(parents=True, exist_ok=True)


def win_to_wsl(path: Path):
    """Преобразует Windows-путь в формат WSL для передачи в rsync."""

    return "/mnt/c" + str(path)[2:].replace("\\", "/")


def run_command(command, log_file: Path, name: str):
    """Запускает внешнюю команду, печатает её в консоль и пишет stdout/stderr в лог."""

    print(f"[{get_timestamp()}] {name}")
    print("Команда:", " ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    with open(log_file, "a", encoding="utf-8") as f:

        f.write(f"\n[{get_timestamp()}] --- {name} ---\n")

        if result.stdout:
            f.write(result.stdout)

        if result.stderr:
            f.write("\nSTDERR:\n")
            f.write(result.stderr)

    return result.returncode


def run_rsync(command, log_file: Path, section):
    """Выполняет rsync-команду и завершает скрипт при критической ошибке."""

    code = run_command(command, log_file, section)

    if code == 0:

        print(GREEN + f"[{get_timestamp()}] OK: {section}" + RESET)

    elif code == 23:

        print(YELLOW +
              f"[{get_timestamp()}] Warning (code 23): {section}" +
              RESET)

    else:

        print(RED +
              f"[{get_timestamp()}] ERROR {code}: {section}" +
              RESET)

        sys.exit(code)


def sync_files():
    """Синхронизирует базы и лог-файлы для всех источников из settings.yaml."""

    for config in sync_configs:

        print("\n" + "=" * 60)

        db_dir = Path(config["db_dir"])
        log_dir = Path(config["log_dir"])

        ensure_dir(db_dir)
        ensure_dir(log_dir)

        log_file = log_dir / "sync.log"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"[{get_timestamp()}] Sync started\n")

        # ---------- DB FILES ----------

        print(f"\n[{get_timestamp()}] Sync DB ({config['name']})")

        rsync_db_cmd = [

            "wsl",
            "rsync",

            "-avz",
            "--progress",

            # SQLite + Google Drive safe
            "--inplace",
            "--partial",
            "--size-only",

            "--no-perms",
            "--no-owner",
            "--no-group",

            "--include=*/",
            "--include=**/*.db",
            "--exclude=*",

            f"{remote_host}:{config['db_remote']}",

            win_to_wsl(db_dir) + "/"
        ]

        run_rsync(rsync_db_cmd, log_file,
                  f"Sync DB: {config['name']}")

        # ---------- LOG FILES ----------

        print(f"\n[{get_timestamp()}] Sync LOG ({config['name']})")

        rsync_log_cmd = [

            "wsl",
            "rsync",

            "-avz",
            "--progress",

            "--inplace",
            "--partial",

            "--no-perms",
            "--no-owner",
            "--no-group",

            f"--include={config['log_pattern']}",
            "--exclude=*",

            f"{remote_host}:{config['log_remote']}",

            win_to_wsl(log_dir) + "/"
        ]

        run_rsync(rsync_log_cmd,
                  log_file,
                  f"Sync LOG: {config['name']}")

    print("\n" + GREEN + "SYNC COMPLETE" + RESET)


if __name__ == "__main__":
    sync_files()
