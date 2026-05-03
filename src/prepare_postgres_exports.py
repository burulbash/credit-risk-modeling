from pathlib import Path
import argparse
import re
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"


def parse_integer_columns_from_schema(schema_path: Path) -> dict[str, list[str]]:
    """
    Reads schema.sql and extracts columns that must be loaded as PostgreSQL integers.
    Handles BIGINT, INTEGER and SMALLINT.
    """
    schema_text = schema_path.read_text(encoding="utf-8")

    table_integer_columns = {}
    current_table = None

    for raw_line in schema_text.splitlines():
        line = raw_line.strip()

        create_match = re.match(r"CREATE TABLE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", line, re.IGNORECASE)
        if create_match:
            current_table = create_match.group(1)
            table_integer_columns[current_table] = []
            continue

        if current_table and line.startswith(");"):
            current_table = None
            continue

        if current_table is None:
            continue

        col_match = re.match(
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s+(BIGINT|INTEGER|SMALLINT)\b",
            line,
            re.IGNORECASE,
        )
        if col_match:
            column_name = col_match.group(1)
            table_integer_columns[current_table].append(column_name)

    return table_integer_columns


def convert_csv_file(input_path: Path, output_path: Path, integer_columns: list[str], chunksize: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        for col in integer_columns:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")

        chunk.to_csv(
            output_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            na_rep="",
        )

        total_rows += len(chunk)
        first_chunk = False

    print(f"OK: {input_path.name} -> {output_path.name} | rows={total_rows:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/exports_medium")
    parser.add_argument("--output-dir", default="data/raw/exports_medium_pg")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_integer_columns = parse_integer_columns_from_schema(SCHEMA_PATH)

    for input_path in sorted(input_dir.glob("*.csv")):
        output_path = output_dir / input_path.name

        if input_path.name.startswith("_"):
            shutil.copy2(input_path, output_path)
            print(f"COPY: {input_path.name}")
            continue

        table_name = input_path.stem
        integer_columns = table_integer_columns.get(table_name, [])
        convert_csv_file(input_path, output_path, integer_columns, args.chunksize)

    print()
    print(f"PostgreSQL-ready exports saved to: {output_dir}")


if __name__ == "__main__":
    main()
