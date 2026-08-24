"""OPA motion, force, TAMSD, and time-window sensitivity metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory

from .manifest import SystemRecord


@dataclass(frozen=True)
class AlphaFit:
    alpha: float
    intercept: float
    prefactor: float
    r_squared: float
    tau_min: float
    tau_max: float
    n_points: int


def unwrap_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Unwrap Cartesian positions with the minimum image between frames."""
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_frames, 3)")
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("cell must be a nonsingular 3x3 matrix")
    if len(positions) < 2:
        return positions.copy()
    inverse = np.linalg.inv(cell)
    fractional_steps = np.diff(positions, axis=0) @ inverse
    fractional_steps -= np.round(fractional_steps)
    cartesian_steps = fractional_steps @ cell
    result = np.empty_like(positions)
    result[0] = positions[0]
    result[1:] = positions[0] + np.cumsum(cartesian_steps, axis=0)
    return result


def tamsd(positions: np.ndarray, max_lag: int) -> np.ndarray:
    """Return time-averaged MSD for integer lags 1..max_lag."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_frames, 3)")
    if max_lag < 1 or max_lag >= len(positions):
        raise ValueError("max_lag must satisfy 1 <= max_lag < n_frames")
    return np.array(
        [
            np.mean(np.sum((positions[lag:] - positions[:-lag]) ** 2, axis=1))
            for lag in range(1, max_lag + 1)
        ],
        dtype=float,
    )


def fit_anomalous_exponent(
    times: np.ndarray, msd: np.ndarray, min_lag: int = 2
) -> AlphaFit:
    """Fit log(MSD) = alpha*log(tau) + intercept on finite positive points."""
    times = np.asarray(times, dtype=float)
    msd = np.asarray(msd, dtype=float)
    if times.shape != msd.shape or times.ndim != 1:
        raise ValueError("times and msd must be same-length one-dimensional arrays")
    lag_numbers = np.arange(1, len(times) + 1)
    mask = (
        (lag_numbers >= min_lag)
        & np.isfinite(times)
        & np.isfinite(msd)
        & (times > 0)
        & (msd > 0)
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two finite positive fit points are required")
    x = np.log(times[mask])
    y = np.log(msd[mask])
    alpha, intercept = np.polyfit(x, y, 1)
    predicted = alpha * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    selected_times = times[mask]
    return AlphaFit(
        alpha=float(alpha),
        intercept=float(intercept),
        prefactor=float(np.exp(intercept)),
        r_squared=float(r_squared),
        tau_min=float(selected_times[0]),
        tau_max=float(selected_times[-1]),
        n_points=int(mask.sum()),
    )


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - x.mean()
    denominator = float(centered @ centered)
    return float(centered @ (y - y.mean()) / denominator) if denominator else 0.0


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _read_series(record: SystemRecord) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(record.csv_path)
    positions = frame[["OPA_COM_x_A", "OPA_COM_y_A", "OPA_COM_z_A"]].to_numpy(float)
    with Trajectory(str(record.traj_path), mode="r") as trajectory:
        cell = np.asarray(trajectory[0].cell.array, dtype=float)
        trajectory_frames = len(trajectory)
    if trajectory_frames != record.n_frames:
        raise ValueError(
            f"{record.traj_path}: trajectory has {trajectory_frames} frames; "
            f"manifest CSV has {record.n_frames}"
        )
    return frame, unwrap_positions(positions, cell), cell


def _regime(alpha: float) -> str:
    if alpha < 0.8:
        return "subdiffusive/caged"
    if alpha <= 1.2:
        return "near-Brownian"
    return "persistent/superdiffusive"


def compute_core_metrics(record: SystemRecord) -> dict[str, float | str | int]:
    """Compute directly observable and TAMSD-based metrics for one system."""
    frame, positions, _cell = _read_series(record)
    times = frame["time_ps"].to_numpy(float)
    displacement = np.linalg.norm(positions - positions[0], axis=1)
    force = frame["F_OPA_norm_eV_A"].to_numpy(float)
    max_lag = max(2, len(positions) // 3)
    msd = tamsd(positions, max_lag)
    tau = np.arange(1, max_lag + 1, dtype=float) * float(np.median(np.diff(times)))
    alpha_fit = fit_anomalous_exponent(tau, msd)
    legacy_slope = _linear_slope(times, displacement**2)
    return {
        "group": record.group,
        "system": record.system,
        "role": record.role,
        "n_frames": len(frame),
        "time_span_ps": float(times[-1] - times[0]),
        "final_disp_A": float(displacement[-1]),
        "mean_disp_A": float(displacement.mean()),
        "max_disp_A": float(displacement.max()),
        "mean_force_eV_A": float(force.mean()),
        "std_force_eV_A": float(force.std(ddof=0)),
        "max_force_eV_A": float(force.max()),
        "force_disp_pearson": _safe_pearson(force, displacement),
        "legacy_single_origin_d_eff": legacy_slope / 6.0,
        "alpha": alpha_fit.alpha,
        "alpha_fit_r2": alpha_fit.r_squared,
        "alpha_prefactor_A2_ps_alpha": alpha_fit.prefactor,
        "alpha_fit_tau_min_ps": alpha_fit.tau_min,
        "alpha_fit_tau_max_ps": alpha_fit.tau_max,
        "alpha_fit_n_points": alpha_fit.n_points,
        "regime": _regime(alpha_fit.alpha),
    }


def block_sensitivity(record: SystemRecord, blocks: int = 4) -> list[dict[str, float | int | str]]:
    """Estimate time-window sensitivity from contiguous, non-independent blocks."""
    if blocks < 2:
        raise ValueError("blocks must be at least 2")
    frame, positions, _cell = _read_series(record)
    times = frame["time_ps"].to_numpy(float)
    force = frame["F_OPA_norm_eV_A"].to_numpy(float)
    rows: list[dict[str, float | int | str]] = []
    for block_index, indices in enumerate(np.array_split(np.arange(len(frame)), blocks), start=1):
        block_positions = positions[indices]
        block_times = times[indices]
        block_force = force[indices]
        relative = block_positions - block_positions[0]
        rms_displacement = float(np.sqrt(np.mean(np.sum(relative**2, axis=1))))
        alpha = float("nan")
        alpha_r2 = float("nan")
        if len(indices) >= 7:
            max_lag = max(2, len(indices) // 3)
            msd = tamsd(block_positions, max_lag)
            tau = np.arange(1, max_lag + 1) * float(np.median(np.diff(block_times)))
            try:
                fit = fit_anomalous_exponent(tau, msd)
                alpha, alpha_r2 = fit.alpha, fit.r_squared
            except ValueError:
                pass
        rows.append(
            {
                "group": record.group,
                "system": record.system,
                "block": block_index,
                "start_ps": float(block_times[0]),
                "end_ps": float(block_times[-1]),
                "n_frames": len(indices),
                "mean_force_eV_A": float(block_force.mean()),
                "rms_disp_from_block_origin_A": rms_displacement,
                "alpha": alpha,
                "alpha_fit_r2": alpha_r2,
            }
        )
    return rows
