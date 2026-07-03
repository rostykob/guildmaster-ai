"""Profile a CSV file and output summary statistics as JSON."""

from __future__ import annotations

import csv
import json
import sys
from typing import Any


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def profile(path: str) -> dict[str, Any]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {"error": "No header row found"}

        columns: dict[str, dict[str, Any]] = {}
        for col in reader.fieldnames:
            columns[col] = {
                "values": [],
                "missing": 0,
                "unique": set(),
            }

        row_count = 0
        for row in reader:
            row_count += 1
            for col in reader.fieldnames:
                val = row.get(col, "")
                if val == "" or val is None:
                    columns[col]["missing"] += 1
                else:
                    columns[col]["values"].append(val)
                    columns[col]["unique"].add(val)

    report: dict[str, Any] = {
        "file": path,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": {},
    }

    for col, info in columns.items():
        col_report: dict[str, Any] = {
            "missing": info["missing"],
            "unique": len(info["unique"]),
        }

        # Detect numeric columns
        numeric_vals = [v for raw in info["values"] if (v := _try_float(raw)) is not None]
        if numeric_vals and len(numeric_vals) > len(info["values"]) * 0.5:
            col_report["dtype"] = "numeric"
            col_report["mean"] = round(sum(numeric_vals) / len(numeric_vals), 2)
            col_report["min"] = min(numeric_vals)
            col_report["max"] = max(numeric_vals)
        else:
            col_report["dtype"] = "text"

        report["columns"][col] = col_report

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: profile_csv.py <path_to_csv>"}))
        sys.exit(1)
    result = profile(sys.argv[1])
    print(json.dumps(result, indent=2))
