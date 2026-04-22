#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from easyvolcap.utils.colmap_utils import read_points3D_binary as read_points3d_binary


SPARSE_DIRS = (
    Path("colmap_sparse/0"),
    Path("mast3r_sparse/0"),
    Path("mapanything_sparse/0"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count COLMAP points3D entries in known sparse reconstruction folders."
    )
    parser.add_argument(
        "root",
        type=Path,
        help=(
            "Root path containing colmap_sparse/0, mast3r_sparse/0, "
            "and mapanything_sparse/0."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    status = 0

    for sparse_dir in SPARSE_DIRS:
        points_file = root / sparse_dir / "points3D.bin"
        label = f"{sparse_dir.as_posix()}/points3D.bin"

        if not points_file.is_file():
            print(f"{label}: missing {points_file}", file=sys.stderr)
            status = 1
            continue

        points3d = read_points3d_binary(str(points_file))
        print(f"{label}: {len(points3d)} points3D")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
