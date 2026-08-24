"""Build the labelled, publication-oriented three-group result2 report."""

from __future__ import annotations

import argparse
from pathlib import Path

from new_solvent_analysis.publication_reporting import build_result2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--refresh", action="store_true", help="refresh an existing result2 tree"
    )
    args = parser.parse_args()
    outputs = build_result2(args.source_root, args.output_root, refresh=args.refresh)
    print(f"generated={len(outputs)}")
    print(args.output_root / "reports" / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
