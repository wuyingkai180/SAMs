"""Place a liquid solution above a fixed periodic surface with Packmol.

Edit only the CONFIG block for normal use.  Surface coordinates and the
orthorhombic cell are read from a POSCAR/VASP file.  Solvent molecules are
placed between a user-defined surface height and the top of the vacuum layer.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "P": 30.974, "S": 32.06, "F": 18.998, "Cl": 35.45,
    "Br": 79.904, "I": 126.904,
}


@dataclass
class SolventInput:
    role: str
    chemical_name: str
    structure_file: str
    density_g_cm3: float
    volume_fraction: float
    n_molecules: int
    molar_mass_g_mol: float | None = None
    n_atoms: int | None = None


@dataclass
class SoluteInput:
    role: str
    chemical_name: str
    structure_file: str
    n_molecules: int = 1
    # None uses the corresponding boundary of the common solution region.
    z_min_a: float | None = None
    z_max_a: float | None = None
    # Set [x, y, z] to fix one centered solute; None lets Packmol place it.
    fixed_center_a: list[float] | None = None
    n_atoms: int | None = None


@dataclass
class SurfaceSystemInput:
    name: str
    surface_file: str
    # Cartesian z coordinate (angstrom) of the highest surface layer.
    surface_top_z_a: float
    solvents: list[SolventInput]
    solutes: list[SoluteInput] = field(default_factory=list)
    # None means: use the cell z length minus PackmolInput.top_margin_a.
    solution_top_z_a: float | None = None


@dataclass
class PackmolInput:
    tolerance: float = 2.2
    # Packmol >= 20.15.0: check distances through the periodic cell boundaries.
    use_periodic_boundary_conditions: bool = True
    # Normally zero for a periodic surface. Use a margin only for a finite cluster.
    xy_margin_a: float = 0.0
    surface_gap_a: float = 2.0
    top_margin_a: float = 1.0
    generated_structure_dir: str = "surface_source_xyz"
    output_dir: str = "surface_systems"
    packmol_input_template: str = "{name}.inp"
    output_xyz_template: str = "{name}.xyz"
    output_vasp_template: str = "{name}.vasp"


@dataclass
class RuntimeInput:
    packmol_executable: str = "packmol"
    run_packmol: bool = True
    skip_existing: bool = False
    quiet: bool = True
    max_atoms: int | None = 30_000


@dataclass
class WorkflowInput:
    workdir: str = "."
    packmol: PackmolInput = field(default_factory=PackmolInput)
    runtime: RuntimeInput = field(default_factory=RuntimeInput)
    systems: list[SurfaceSystemInput] = field(default_factory=list)


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


# =========================
# editable configuration
# =========================
CONFIG = WorkflowInput(
    workdir=str(SCRIPT_DIR),
    packmol=PackmolInput(
        tolerance=2.2,
        use_periodic_boundary_conditions=True,
        xy_margin_a=0.0,
        surface_gap_a=2.0,
        top_margin_a=1.0,
    ),
    runtime=RuntimeInput(
        packmol_executable="packmol",
        # false tests the complete input-generation flow without launching Packmol.
        run_packmol=True,
        skip_existing=False,
        quiet=True,
        max_atoms=None,
    ),
    systems=[
        SurfaceSystemInput(
            name="surface_solution_ex1",
            surface_file="surface.vasp",
            # highest atom/layer in the supplied surface.vasp.
            surface_top_z_a=11.0115,
            # keep 15 A vacuum above the liquid to separate z-periodic images.
            solution_top_z_a=35.0,
            solvents=[
                SolventInput(
                    role="solvent_1",
                    chemical_name="acetone",
                    structure_file="Actone.vasp",
                    density_g_cm3=0.7845,
                    volume_fraction=0.10,
                    n_molecules=7,
                ),
                SolventInput(
                    role="solvent_2",
                    chemical_name="n-heptane",
                    structure_file="n-heptane.vasp",
                    density_g_cm3=0.684,
                    volume_fraction=0.90,
                    n_molecules=33,
                ),
            ],
            # solutes use the common solution box unless z_min_a/z_max_a are set.
            solutes=[
                SoluteInput(
                    role="solute_1",
                    chemical_name="OPA",
                    structure_file="solute_1.vasp",
                    # set to 0 to retain this interface without adding the solute.
                    n_molecules=0,
                    z_min_a=15.0,
                    z_max_a=30.0,
                    fixed_center_a=None,
                ),
            ],
        ),
    ],
)


def _path(workdir: str | Path, filename: str | Path) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else Path(workdir) / path


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")


def read_xyz_structure(xyz_file: str | Path) -> tuple[list[str], np.ndarray]:
    lines = Path(xyz_file).read_text(encoding="utf-8").splitlines()
    n_atoms = int(lines[0].strip())
    if len(lines) < n_atoms + 2:
        raise ValueError(f"Incomplete XYZ file: {xyz_file}")
    symbols, coords = [], []
    for line in lines[2 : 2 + n_atoms]:
        fields = line.split()
        symbols.append(fields[0])
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return symbols, np.asarray(coords, dtype=float)


def read_vasp_structure(
    vasp_file: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    lines = Path(vasp_file).read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError(f"{vasp_file} does not look like a POSCAR/VASP file.")
    scale = float(lines[1].split()[0])
    if scale <= 0:
        raise ValueError("Only a positive POSCAR scale factor is supported.")
    cell = np.asarray(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
        dtype=float,
    ) * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(symbols) != len(counts):
        raise ValueError(f"Element/count mismatch in {vasp_file}")
    expanded = [symbol for symbol, count in zip(symbols, counts) for _ in range(count)]
    start = 7
    if lines[start].strip().lower().startswith("s"):
        start += 1
    mode = lines[start].strip().lower()
    start += 1
    coords = np.asarray(
        [[float(x) for x in lines[start + i].split()[:3]] for i in range(len(expanded))],
        dtype=float,
    )
    coords = coords @ cell if mode.startswith(("d", "f")) else coords * scale
    return expanded, coords, cell


def read_structure(structure_file: str | Path) -> tuple[list[str], np.ndarray]:
    if Path(structure_file).suffix.lower() == ".xyz":
        return read_xyz_structure(structure_file)
    symbols, coords, _ = read_vasp_structure(structure_file)
    return symbols, coords


def save_xyz(path: str | Path, symbols: list[str], coords: np.ndarray, comment: str) -> Path:
    path = Path(path)
    lines = [str(len(symbols)), comment]
    lines.extend(
        f"{symbol} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
        for symbol, xyz in zip(symbols, coords)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_vasp(
    path: str | Path, symbols: list[str], coords: np.ndarray, cell: np.ndarray
) -> Path:
    """Save Packmol output as Cartesian POSCAR while retaining the surface cell."""
    order: list[str] = []
    for symbol in symbols:
        if symbol not in order:
            order.append(symbol)
    grouped_indices = [i for symbol in order for i, value in enumerate(symbols) if value == symbol]
    counts = [symbols.count(symbol) for symbol in order]
    lines = ["Surface + solution generated by Packmol", "1.0"]
    lines.extend("  " + "  ".join(f"{value:.12f}" for value in vector) for vector in cell)
    lines.extend(["  " + "  ".join(order), "  " + "  ".join(map(str, counts)), "Cartesian"])
    lines.extend(
        "  " + "  ".join(f"{value:.10f}" for value in coords[i])
        for i in grouped_indices
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def molecular_mass_g_mol(symbols: list[str]) -> float:
    missing = sorted({symbol for symbol in symbols if symbol not in ATOMIC_MASSES})
    if missing:
        raise ValueError(f"Missing atomic masses for: {', '.join(missing)}")
    return float(sum(ATOMIC_MASSES[symbol] for symbol in symbols))


def _orthorhombic_lengths(cell: np.ndarray, surface_file: Path) -> np.ndarray:
    off_diagonal = cell - np.diag(np.diag(cell))
    if not np.allclose(off_diagonal, 0.0, atol=1e-8) or np.any(np.diag(cell) <= 0):
        raise ValueError(
            f"{surface_file} must use positive, axis-aligned orthorhombic lattice vectors "
            "because Packmol's inside-box region is Cartesian."
        )
    return np.diag(cell).copy()


def _region(
    config: WorkflowInput, system: SurfaceSystemInput
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface_path = _path(config.workdir, system.surface_file)
    _, surface_coords, cell = read_vasp_structure(surface_path)
    lengths = _orthorhombic_lengths(cell, surface_path)
    packmol = config.packmol
    lower = np.array(
        [packmol.xy_margin_a, packmol.xy_margin_a,
         system.surface_top_z_a + packmol.surface_gap_a]
    )
    upper_z = (
        system.solution_top_z_a
        if system.solution_top_z_a is not None
        else lengths[2] - packmol.top_margin_a
    )
    upper = np.array([lengths[0] - packmol.xy_margin_a,
                      lengths[1] - packmol.xy_margin_a, upper_z])
    if np.any(lower >= upper):
        raise ValueError(f"Invalid solution region for {system.name}: {lower} -> {upper}")
    actual_top = float(surface_coords[:, 2].max())
    actual_bottom = float(surface_coords[:, 2].min())
    if actual_bottom < -1e-6 or actual_top > lengths[2] + 1e-6:
        raise ValueError(
            f"Surface z coordinates for {system.name} must lie inside 0..{lengths[2]:g} A; "
            f"found {actual_bottom:.6f}..{actual_top:.6f} A. Recenter the slab in z."
        )
    if system.surface_top_z_a + 1e-6 < actual_top:
        raise ValueError(
            f"surface_top_z_a={system.surface_top_z_a:g} A for {system.name} is below "
            f"the highest surface atom ({actual_top:.6f} A)."
        )
    return lower, upper, cell


def _generated_xyz(
    config: WorkflowInput, source_file: str | Path, output_stem: str
) -> Path:
    source = _path(config.workdir, source_file)
    output_dir = _path(config.workdir, config.packmol.generated_structure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{output_stem}.xyz"
    symbols, coords = read_structure(source)
    return save_xyz(output, symbols, coords, f"Converted from {source.name}")


def _centered_xyz(
    config: WorkflowInput,
    source_file: str | Path,
    output_stem: str,
    center_a: list[float],
) -> Path:
    source = _path(config.workdir, source_file)
    symbols, coords = read_structure(source)
    coords = coords - coords.mean(axis=0) + np.asarray(center_a, dtype=float)
    output_dir = _path(config.workdir, config.packmol.generated_structure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return save_xyz(
        output_dir / f"{output_stem}.xyz",
        symbols,
        coords,
        f"Converted and centered from {source.name}",
    )


def _surface_xyz(
    config: WorkflowInput, system: SurfaceSystemInput, output_stem: str
) -> Path:
    """Write the fixed slab with x/y coordinates wrapped into its periodic cell."""
    source = _path(config.workdir, system.surface_file)
    symbols, coords, cell = read_vasp_structure(source)
    lengths = _orthorhombic_lengths(cell, source)
    coords[:, 0] %= lengths[0]
    coords[:, 1] %= lengths[1]
    output_dir = _path(config.workdir, config.packmol.generated_structure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return save_xyz(
        output_dir / f"{output_stem}.xyz",
        symbols,
        coords,
        f"Converted and wrapped in x/y from {source.name}",
    )


def _ref(config: WorkflowInput, path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(config.workdir).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _output_path(config: WorkflowInput, system: SurfaceSystemInput, template: str) -> Path:
    return _path(config.workdir, config.packmol.output_dir) / template.format(
        name=_safe_name(system.name)
    )


def prepare_system(config: WorkflowInput, system: SurfaceSystemInput) -> None:
    surface_path = _path(config.workdir, system.surface_file)
    if not surface_path.is_file():
        raise FileNotFoundError(f"Missing surface structure: {surface_path}")
    _region(config, system)
    if not system.solvents:
        raise ValueError(f"{system.name} has no solvents.")
    if not np.isclose(sum(s.volume_fraction for s in system.solvents), 1.0):
        raise ValueError(f"Solvent volume fractions for {system.name} must sum to 1.0.")
    for solvent in system.solvents:
        source = _path(config.workdir, solvent.structure_file)
        if not source.is_file():
            raise FileNotFoundError(f"Missing {solvent.role} structure: {source}")
        if solvent.n_molecules < 1:
            raise ValueError(f"{solvent.role}.n_molecules must be positive.")
        symbols, _ = read_structure(source)
        solvent.n_atoms = len(symbols)
        solvent.molar_mass_g_mol = molecular_mass_g_mol(symbols)
    lower, upper, _ = _region(config, system)
    for solute in system.solutes:
        if solute.n_molecules < 0:
            raise ValueError(f"{solute.role}.n_molecules cannot be negative.")
        if solute.n_molecules == 0:
            solute.n_atoms = 0
            continue
        source = _path(config.workdir, solute.structure_file)
        if not source.is_file():
            raise FileNotFoundError(f"Missing {solute.role} structure: {source}")
        if solute.fixed_center_a is not None and solute.n_molecules != 1:
            raise ValueError(f"Fixed {solute.role} must have n_molecules=1.")
        z_min = lower[2] if solute.z_min_a is None else solute.z_min_a
        z_max = upper[2] if solute.z_max_a is None else solute.z_max_a
        if z_min < lower[2] or z_max > upper[2] or z_min >= z_max:
            raise ValueError(
                f"{solute.role} z range {z_min:g}..{z_max:g} A must lie inside "
                f"the solution range {lower[2]:g}..{upper[2]:g} A."
            )
        if solute.fixed_center_a is not None:
            center = np.asarray(solute.fixed_center_a, dtype=float)
            if center.shape != (3,) or np.any(center < lower) or np.any(center > upper):
                raise ValueError(
                    f"Fixed center for {solute.role} must lie inside the solution box."
                )
        symbols, _ = read_structure(source)
        solute.n_atoms = len(symbols)


def expected_atom_count(config: WorkflowInput, system: SurfaceSystemInput) -> int:
    surface_symbols, _, _ = read_vasp_structure(_path(config.workdir, system.surface_file))
    return len(surface_symbols) + sum(
        int(solvent.n_atoms or 0) * solvent.n_molecules for solvent in system.solvents
    ) + sum(
        int(solute.n_atoms or 0) * solute.n_molecules for solute in system.solutes
    )


def actual_volume_fraction(solvent: SolventInput, system: SurfaceSystemInput) -> float:
    volumes = [
        item.n_molecules * float(item.molar_mass_g_mol) / item.density_g_cm3
        for item in system.solvents
    ]
    return float(volumes[system.solvents.index(solvent)] / sum(volumes))


def write_packmol_input(config: WorkflowInput, system: SurfaceSystemInput) -> Path:
    prepare_system(config, system)
    lower, upper, cell = _region(config, system)
    lengths = np.diag(cell)
    output_dir = _path(config.workdir, config.packmol.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(system.name)
    surface_xyz = _surface_xyz(config, system, f"{stem}_surface_fixed")
    output_xyz = _output_path(config, system, config.packmol.output_xyz_template)
    header_lines = [
        f"tolerance {config.packmol.tolerance}",
        "filetype xyz",
        f"output {_ref(config, output_xyz)}",
    ]
    if config.packmol.use_periodic_boundary_conditions:
        header_lines.append("pbc " + " ".join(f"{value:.8f}" for value in lengths))
    blocks = [
        "\n".join(header_lines),
        f"structure {_ref(config, surface_xyz)}\n  number 1\n"
        "  fixed 0. 0. 0. 0. 0. 0.\nend structure",
    ]
    for solute in system.solutes:
        if solute.n_molecules == 0:
            continue
        if solute.fixed_center_a is not None:
            solute_xyz = _centered_xyz(
                config,
                solute.structure_file,
                f"{stem}_{solute.role}_fixed",
                solute.fixed_center_a,
            )
            blocks.append(
                f"structure {_ref(config, solute_xyz)}\n"
                "  number 1\n  fixed 0. 0. 0. 0. 0. 0.\nend structure"
            )
        else:
            solute_xyz = _generated_xyz(
                config, solute.structure_file, f"{stem}_{solute.role}"
            )
            z_min = lower[2] if solute.z_min_a is None else solute.z_min_a
            z_max = upper[2] if solute.z_max_a is None else solute.z_max_a
            solute_box = " ".join(
                f"{value:.8f}"
                for value in [lower[0], lower[1], z_min, upper[0], upper[1], z_max]
            )
            blocks.append(
                f"structure {_ref(config, solute_xyz)}\n"
                f"  number {solute.n_molecules}\n"
                f"  inside box {solute_box}\nend structure"
            )
    box = " ".join(f"{value:.8f}" for value in [*lower, *upper])
    for solvent in system.solvents:
        solvent_xyz = _generated_xyz(
            config, solvent.structure_file, f"{stem}_{solvent.role}"
        )
        blocks.append(
            f"structure {_ref(config, solvent_xyz)}\n"
            f"  number {solvent.n_molecules}\n  inside box {box}\nend structure"
        )
    input_path = _output_path(config, system, config.packmol.packmol_input_template)
    input_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return input_path


def resolve_packmol_executable(executable: str) -> str:
    requested = os.environ.get("PACKMOL_EXECUTABLE", executable)
    requested_path = Path(requested).expanduser()
    if requested_path.is_file():
        return str(requested_path.resolve())
    resolved = shutil.which(requested)
    if resolved:
        return resolved
    candidates = [
        Path(sys.executable).resolve().parent / "packmol.exe",
        Path(sys.prefix) / "Scripts" / "packmol.exe",
        Path(sys.prefix) / "Library" / "bin" / "packmol.exe",
        Path(sys.prefix) / "bin" / "packmol",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"Packmol executable was not found for {requested!r}. Set "
        "CONFIG.runtime.packmol_executable or PACKMOL_EXECUTABLE."
    )


def validate_and_convert_output(
    config: WorkflowInput, system: SurfaceSystemInput
) -> tuple[Path, Path]:
    xyz_path = _output_path(config, system, config.packmol.output_xyz_template)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Packmol did not create {xyz_path}")
    symbols, coords = read_xyz_structure(xyz_path)
    expected = expected_atom_count(config, system)
    if len(symbols) != expected:
        raise ValueError(
            f"Atom-count mismatch for {system.name}: expected {expected}, found {len(symbols)}"
        )
    _, _, cell = read_vasp_structure(_path(config.workdir, system.surface_file))
    vasp_path = _output_path(config, system, config.packmol.output_vasp_template)
    save_vasp(vasp_path, symbols, coords, cell)
    return xyz_path, vasp_path


def run_packmol(config: WorkflowInput) -> list[tuple[Path, Path]]:
    executable = resolve_packmol_executable(config.runtime.packmol_executable)
    print(f"Resolved Packmol executable: {executable}")
    outputs = []
    for system in config.systems:
        input_path = write_packmol_input(config, system)
        xyz_path = _output_path(config, system, config.packmol.output_xyz_template)
        if config.runtime.skip_existing and xyz_path.exists():
            outputs.append(validate_and_convert_output(config, system))
            print(f"Skipping existing output: {xyz_path}")
            continue
        log_path = input_path.with_suffix(".log")
        print(f"Running Packmol for {system.name}; log: {log_path}")
        with input_path.open("rb") as stdin:
            if config.runtime.quiet:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [executable], cwd=config.workdir, stdin=stdin,
                        stdout=log, stderr=subprocess.STDOUT, check=True,
                    )
            else:
                subprocess.run([executable], cwd=config.workdir, stdin=stdin, check=True)
        outputs.append(validate_and_convert_output(config, system))
    return outputs


def main(config: WorkflowInput) -> None:
    input_paths = []
    for system in config.systems:
        prepare_system(config, system)
        atoms = expected_atom_count(config, system)
        if config.runtime.max_atoms is not None and atoms > config.runtime.max_atoms:
            raise ValueError(f"{system.name} exceeds max_atoms: {atoms:,}")
        input_paths.append(write_packmol_input(config, system))
        lower, upper, _ = _region(config, system)
        composition = ", ".join(
            f"{s.role} [{s.chemical_name}]={s.n_molecules} "
            f"(actual_v={100 * actual_volume_fraction(s, system):.3f}%)"
            for s in system.solvents
        )
        solutes = ", ".join(
            f"{s.role} [{s.chemical_name}]={s.n_molecules}" for s in system.solutes
        ) or "none"
        print(
            f"{system.name}: solution box z={lower[2]:g}..{upper[2]:g} A; "
            f"{composition}; solutes: {solutes}; atoms={atoms:,}"
        )
    config_path = _path(config.workdir, "surface_solution_packmol_config.json")
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    print("Generated Packmol inputs:")
    for path in input_paths:
        print(" ", path)
    if config.runtime.run_packmol:
        for xyz_path, vasp_path in run_packmol(config):
            print(f"Generated and verified: {xyz_path}; {vasp_path}")


if __name__ == "__main__":
    main(CONFIG)
