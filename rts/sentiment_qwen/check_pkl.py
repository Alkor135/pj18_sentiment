from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd


def load_pickle_as_dataframe(path: Path) -> pd.DataFrame:
    with path.open("rb") as file:
        data = pickle.load(file)

    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def print_preview(path: Path) -> None:
    df = load_pickle_as_dataframe(path)

    print("=" * 100)
    print(f"File: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    with pd.option_context(
        "display.width", 160,
        "display.max_columns", 50,
        "display.max_colwidth", 80,
        "display.float_format", "{:,.4f}".format,
    ):
        print()
        print("HEAD 5")
        print(df.head(5))
        print()
        print("TAIL 5")
        print(df.tail(5))
        print()


def resolve_directory(args: list[str]) -> Path:
    if args:
        return Path(args[0]).resolve()
    return Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    directory = resolve_directory(args)

    pkl_files = sorted(directory.glob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in: {directory}")
        return 1

    for pkl_file in pkl_files:
        print_preview(pkl_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
