"""Command-line entry point for the corrected three-group OPA solvent analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from new_solvent_analysis.pipeline import run_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--literature", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(
        args.data_root, args.output_root, args.literature,
        dry_run=args.dry_run, resume=args.resume
    )
    print(
        f"status={summary.status}; systems={summary.system_count}; "
        f"trash_systems={summary.excluded_trash_systems}"
    )
    for path in summary.report_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
