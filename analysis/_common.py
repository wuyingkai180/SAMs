"""
Shared helpers for the SAMs/analysis/*.py post-hoc analysis scripts.

Only genuinely duplicated code lives here:
  - `pearson()` was defined twice, byte-for-byte equivalent (same formula,
    just a different way of computing the mean), in analyze_pure_solvents.py
    and build_combined_report.py.
  - MUTED / GRID / TEXT_PRIMARY / TEXT_SECONDARY were the same literal SVG
    color values repeated in analyze_pure_solvents.py,
    analyze_opa_displacement_dynamics.py, and analyze_opa_solvation_structure.py.

Things that only *looked* similar were deliberately left where they were
instead of being forced into a shared abstraction here, e.g.:
  - `fmt()` in analyze_pure_solvents.py (nd=3 default, no NaN handling) vs.
    analyze_opa_displacement_dynamics.py (nd=4 default, formats NaN as "n/a")
    -- different defaults and different NaN behavior.
  - hbar_chart_svg (analyze_pure_solvents.py, always-positive bars) vs.
    bar_svg (analyze_opa_displacement_dynamics.py, zero-centered bars that
    support negative values) -- different chart semantics, not the same
    function with cosmetic differences.
  - line_chart_svg (analyze_pure_solvents.py, single series vs. time) vs.
    line_svg (analyze_opa_solvation_structure.py, two series vs. r) --
    different signatures and different plots.
"""

from __future__ import annotations

import math

# SVG chart palette shared by the scripts that draw their own inline charts.
MUTED = "#898781"
GRID = "#e1e0d9"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient; 0.0 if either series has zero variance."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx and deny else 0.0
