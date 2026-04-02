#!/usr/bin/env python3
"""Build a Markdown table from summary metrics in data/result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PREFERRED_KEY_ORDER = [
    "psnr_mean",
    "psnr_std",
    "ssim_mean",
    "ssim_std",
    "lpips_mean",
    "lpips_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan dataset results and build a Markdown table from each file's "
            "`summary` section."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/result"),
        help="Root folder that contains dataset result folders (default: data/result).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/result/summary_table.md"),
        help="Output Markdown file path. Use '-' to write to stdout.",
    )
    return parser.parse_args()


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if value is None:
        return ""
    return str(value)


def find_metric_files(results_dir: Path) -> list[Path]:
    metric_files: list[Path] = []
    for name in ("metric.json", "metrics.json"):
        metric_files.extend(results_dir.rglob(name))
    return sorted(set(metric_files))


def collect_rows(results_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    key_set: set[str] = set()

    for metric_file in find_metric_files(results_dir):
        try:
            data = json.loads(metric_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Failed to parse {metric_file}: {exc}", file=sys.stderr)
            continue

        summary = data.get("summary")
        if not isinstance(summary, dict):
            print(f"[WARN] Missing or invalid `summary` in {metric_file}", file=sys.stderr)
            continue

        dataset = str(metric_file.parent.relative_to(results_dir))
        row = {"dataset": dataset, **summary}
        rows.append(row)
        key_set.update(summary.keys())

    preferred = [k for k in PREFERRED_KEY_ORDER if k in key_set]
    remaining = sorted(k for k in key_set if k not in PREFERRED_KEY_ORDER)
    ordered_keys = preferred + remaining
    rows.sort(key=lambda item: item["dataset"])
    return rows, ordered_keys


def build_markdown(rows: list[dict[str, Any]], keys: list[str]) -> str:
    headers = ["dataset", *keys]
    table_rows = [[header for header in headers]]
    for row in rows:
        table_rows.append([format_value(row.get(header, "")) for header in headers])

    col_widths = [
        max(len(table_rows[row_idx][col_idx]) for row_idx in range(len(table_rows)))
        for col_idx in range(len(headers))
    ]

    def format_row(cells: list[str]) -> str:
        padded = [f"{cell:<{col_widths[idx]}}" for idx, cell in enumerate(cells)]
        return f"| {' | '.join(padded)} |"

    separator_cells = ["-" * width for width in col_widths]
    lines = [format_row(table_rows[0]), format_row(separator_cells)]
    for cells in table_rows[1:]:
        lines.append(format_row(cells))

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir

    if not results_dir.exists() or not results_dir.is_dir():
        print(f"[ERROR] Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    rows, keys = collect_rows(results_dir)
    if not rows:
        print(f"[ERROR] No valid metric files found in: {results_dir}", file=sys.stderr)
        return 1

    markdown = build_markdown(rows, keys)

    if str(args.output) == "-":
        print(markdown, end="")
        return 0

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        print(f"[ERROR] Failed to write {args.output}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} dataset summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
