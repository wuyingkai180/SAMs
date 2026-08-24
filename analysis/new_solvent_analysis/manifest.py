"""Input discovery and validation for the corrected solvent datasets."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path


REQUIRED_COLUMNS = (
    "time_ps",
    "OPA_COM_x_A",
    "OPA_COM_y_A",
    "OPA_COM_z_A",
    "OPA_disp_norm_A",
    "F_OPA_norm_eV_A",
)

GROUP3_METADATA = {
    "acetone_5_n-heptane_95": (
        "n-heptane-base-composition", "acetone", "n-heptane", 0.05
    ),
    "isopropanol_5_n-heptane_95": (
        "n-heptane-base-composition", "isopropanol", "n-heptane", 0.05
    ),
    "thf_n-heptane": (
        "n-heptane-base-composition", "thf", "n-heptane", 0.05
    ),
    "toluene_n-heptane": (
        "n-heptane-base-composition", "toluene", "n-heptane", 0.05
    ),
    "thf_toluene": (
        "alternate-base-composition", "thf", "toluene", 0.05
    ),
}


class ManifestError(ValueError):
    """Raised when an input system violates a required invariant."""


@dataclass(frozen=True)
class SystemRecord:
    group: str
    system: str
    role: str
    csv_path: Path
    traj_path: Path
    xyz_path: Path
    file_prefix: str
    cosolvent: str | None
    base_solvent: str
    cosolvent_fraction: float | None
    n_frames: int
    time_start_ps: float
    time_end_ps: float
    sha256: str

    @property
    def components(self) -> tuple[str, ...]:
        if self.cosolvent is None:
            return (self.base_solvent,)
        return (self.cosolvent, self.base_solvent)


def _one(path: Path, pattern: str, label: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ManifestError(
            f"{path}: expected exactly one {label} matching {pattern!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _csv_metadata(path: Path) -> tuple[int, float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"{path}: missing required columns {missing}")
        times = []
        for line_number, row in enumerate(reader, start=2):
            try:
                times.append(float(row["time_ps"]))
            except (TypeError, ValueError) as exc:
                raise ManifestError(
                    f"{path}:{line_number}: invalid time_ps {row.get('time_ps')!r}"
                ) from exc
    if not times:
        raise ManifestError(f"{path}: CSV has no data rows")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ManifestError(f"{path}: time_ps must be strictly increasing")
    if len(times) != 101 or times[0] != 0.0 or times[-1] != 10.0:
        raise ManifestError(
            f"{path}: expected 101 frames spanning 0.0-10.0 ps; "
            f"found {len(times)} spanning {times[0]}-{times[-1]}"
        )
    return len(times), times[0], times[-1]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification(group: str, system: str) -> tuple[str, str | None, str, float | None]:
    if group == "group1":
        match = re.fullmatch(r"acetone_(\d+)_n-heptane_(\d+)", system)
        if not match or int(match.group(1)) + int(match.group(2)) != 100:
            raise ManifestError(f"{group}/{system}: invalid acetone concentration name")
        fraction = int(match.group(1)) / 100.0
        return "concentration-series", "acetone", "n-heptane", fraction
    if group == "group2":
        return "pure-solvent", None, system, None
    try:
        return GROUP3_METADATA[system]
    except KeyError as exc:
        raise ManifestError(f"group3/{system}: no approved comparison role") from exc


def build_manifest(data_root: Path) -> list[SystemRecord]:
    """Discover and strictly validate group1/group2/group3 inputs."""
    data_root = Path(data_root)
    records: list[SystemRecord] = []
    for group in ("group1", "group2", "group3"):
        group_path = data_root / group
        if not group_path.is_dir():
            raise ManifestError(f"Missing required group directory: {group_path}")
        for system_path in sorted(path for path in group_path.iterdir() if path.is_dir()):
            csv_path = _one(system_path, "*_opa_motion_force.csv", "motion/force CSV")
            traj_path = _one(system_path, "*_md_300K.traj", "ASE trajectory")
            xyz_path = _one(system_path, "*_final_300K.xyz", "final XYZ")
            file_prefix = csv_path.name.removesuffix("_opa_motion_force.csv")
            expected_traj = system_path / f"{file_prefix}_md_300K.traj"
            expected_xyz = system_path / f"{file_prefix}_final_300K.xyz"
            if traj_path != expected_traj or xyz_path != expected_xyz:
                raise ManifestError(
                    f"{system_path}: CSV/trajectory/XYZ prefixes disagree "
                    f"({csv_path.name}, {traj_path.name}, {xyz_path.name})"
                )
            n_frames, time_start, time_end = _csv_metadata(csv_path)
            role, cosolvent, base_solvent, fraction = _classification(group, system_path.name)
            records.append(
                SystemRecord(
                    group=group,
                    system=system_path.name,
                    role=role,
                    csv_path=csv_path,
                    traj_path=traj_path,
                    xyz_path=xyz_path,
                    file_prefix=file_prefix,
                    cosolvent=cosolvent,
                    base_solvent=base_solvent,
                    cosolvent_fraction=fraction,
                    n_frames=n_frames,
                    time_start_ps=time_start,
                    time_end_ps=time_end,
                    sha256=_hash_file(csv_path),
                )
            )
    return records


def write_manifest(records: list[SystemRecord], path: Path) -> None:
    """Write a portable manifest with paths relative to the current project."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in SystemRecord.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            for key in ("csv_path", "traj_path", "xyz_path"):
                row[key] = Path(row[key]).as_posix()
            writer.writerow(row)
