"""
Проверка актуальности скачанных rss-БД: количество новостей
по каждому провайдеру за текущую дату. Простой вывод в консоль.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
import yaml

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def load_config():
    with open(Path(__file__).parent / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_by_provider(db_path: Path, src: dict, day: str):
    date_col = src["date_column"]
    prov_col = src.get("provider_column")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    if prov_col:
        rows = cur.execute(
            f"SELECT {prov_col}, COUNT(*) FROM news "
            f"WHERE {date_col} LIKE ? GROUP BY {prov_col} ORDER BY {prov_col}",
            (f"{day}%",),
        ).fetchall()
    else:
        (total,) = cur.execute(
            f"SELECT COUNT(*) FROM news WHERE {date_col} LIKE ?",
            (f"{day}%",),
        ).fetchone()
        rows = [(src["provider_fixed"], total)]

    conn.close()
    return rows


def main():
    cfg = load_config()
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now()

    print(f"\n=== RSS DB check: {today} ===\n")

    grand_total = 0
    any_missing = False

    for src in cfg["sources"]:
        db_dir = Path(src["db_dir"])
        db_name = src["db_file_pattern"].format(year=now.year, month=now.month)
        db_path = db_dir / db_name

        print(f"[{src['name']}] {db_path.name}")

        if not db_path.exists():
            print(RED + f"  файл не найден: {db_path}" + RESET)
            any_missing = True
            continue

        rows = count_by_provider(db_path, src, today)
        src_total = sum(n for _, n in rows)
        grand_total += src_total

        if not rows or src_total == 0:
            print(YELLOW + "  новостей за сегодня нет" + RESET)
            continue

        for prov, n in rows:
            print(f"  {prov:<12}: {n:>5}")
        print(f"  {'ИТОГО':<12}: {src_total:>5}")
        print()

    color = GREEN if grand_total > 0 and not any_missing else YELLOW
    print(color + f"ВСЕГО за {today}: {grand_total}" + RESET)

    return 0 if grand_total > 0 and not any_missing else 1


if __name__ == "__main__":
    sys.exit(main())
