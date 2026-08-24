"""Reproducible analysis of the corrected three-group OPA solvent dataset."""

from __future__ import annotations

__version__ = "1.0.0"


def run_analysis(*args, **kwargs):
    """Import the pipeline lazily so lightweight modules stay importable."""
    from .pipeline import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)


__all__ = ["__version__", "run_analysis"]

