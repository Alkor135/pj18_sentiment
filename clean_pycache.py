import shutil
from pathlib import Path

ROOT = Path(__file__).parent


SKIP_DIRS = {".venv", ".git"}


def main() -> None:
    removed = 0
    skipped = 0
    for path in ROOT.rglob("__pycache__"):
        if not path.is_dir():
            continue
        rel = path.relative_to(ROOT)
        if SKIP_DIRS & set(rel.parts):
            print(f"skip: {rel}")
            skipped += 1
            continue
        shutil.rmtree(path)
        print(rel)
        removed += 1
    print(f"Removed: {removed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
