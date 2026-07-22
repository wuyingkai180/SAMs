"""Build pure-solvent, binary, and ternary solute/solvent systems with Packmol."""

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


AVOGADRO = 6.02214076e23
PFP_NEIGHBOR_DENSITY_FACTOR = 1_663.847198
ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.06,
    "F": 18.998,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
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
class SystemInput:
    name: str
    box_length_a: float
    solvents: list[SolventInput]
    n_solute_1_molecules: int = 1


@dataclass
class PackmolInput:
    margin: float = 2.0
    tolerance: float = 2.2
    fixed_single_solute_1: bool = True
    generated_structure_dir: str = "packmol_source_xyz"
    output_dir: str = "packed_systems"
    packmol_input_template: str = "{name}.inp"
    output_xyz_template: str = "{name}.xyz"


@dataclass
class RuntimeInput:
    packmol_executable: str = "packmol"
    run_packmol: bool = True
    skip_existing: bool = False
    quiet: bool = True
    max_atoms: int | None = 30_000
    max_estimated_neighbors: int | None = 1_650_000


def _solvent(
    role: str,
    chemical_name: str,
    structure_file: str,
    density: float,
    volume_fraction: float,
    n_molecules: int,
) -> SolventInput:
    return SolventInput(
        role=role,
        chemical_name=chemical_name,
        structure_file=structure_file,
        density_g_cm3=density,
        volume_fraction=volume_fraction,
        n_molecules=n_molecules,
    )


@dataclass
class WorkflowInput:
    workdir: str = "."
    solute_1_file: str = "solute_1.vasp"
    packmol: PackmolInput = field(default_factory=PackmolInput)
    runtime: RuntimeInput = field(default_factory=RuntimeInput)
    systems: list[SystemInput] = field(default_factory=list)
    n_solute_1_atoms: int | None = None


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


# =========================
# Editable configuration
# =========================
CONFIG = WorkflowInput(
    # Input and output paths are resolved from the directory containing this script.
    workdir=str(SCRIPT_DIR),
    solute_1_file="solute_1.vasp",
    packmol=PackmolInput(
        margin=2.0,
        tolerance=2.2,
        fixed_single_solute_1=True,
        generated_structure_dir="source_xyz",
        output_dir="systems",
        packmol_input_template="{name}.inp",
        output_xyz_template="{name}.xyz",
    ),
    runtime=RuntimeInput(
        # "packmol" uses the executable from the active environment/PATH.
        # An absolute path such as "/home/jovyan/miniconda3/bin/packmol" also works.
        packmol_executable="packmol",
        run_packmol=True,
        skip_existing=False,
        quiet=True,
        max_atoms=30_000,
        max_estimated_neighbors=1_650_000,
    ),
    systems=[
        SystemInput(
            name="ex1",
            box_length_a=47.0,
            n_solute_1_molecules=1,
            solvents=[
                _solvent("solvent_1", "n-heptane", "n-heptane.vasp", 0.684, 1.00, 427),
            ],
        ),
        SystemInput(
            name="ex2",
            box_length_a=48.0,
            n_solute_1_molecules=1,
            solvents=[
                _solvent("solvent_1", "acetone", "Actone.vasp", 0.7845, 0.10, 90),
                _solvent("solvent_2", "n-heptane", "n-heptane.vasp", 0.684, 0.90, 409),
            ],
        ),
        SystemInput(
            name="ex3",
            box_length_a=47.0,
            n_solute_1_molecules=2,
            solvents=[
                _solvent("solvent_1", "acetone", "Actone.vasp", 0.7845, 0.10, 84),
                _solvent("solvent_2", "n-heptane", "n-heptane.vasp", 0.684, 0.80, 341),
                _solvent("solvent_3", "thf", "thf.vasp", 0.889, 0.10, 77),
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
    symbols = []
    coords = []
    for line in lines[2 : 2 + n_atoms]:
        fields = line.split()
        symbols.append(fields[0])
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return symbols, np.asarray(coords, dtype=float)


def read_vasp_structure(vasp_file: str | Path) -> tuple[list[str], np.ndarray]:
    lines = Path(vasp_file).read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError(f"{vasp_file} does not look like a valid POSCAR/VASP file.")

    scale = float(lines[1].split()[0])
    cell = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
        dtype=float,
    )
    cell *= scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]

    coord_start = 7
    if lines[coord_start].strip().lower().startswith("s"):
        coord_start += 1
    coord_mode = lines[coord_start].strip().lower()
    coord_start += 1

    expanded_symbols = []
    for symbol, count in zip(symbols, counts):
        expanded_symbols.extend([symbol] * count)

    coords = np.array(
        [
            [float(x) for x in lines[coord_start + i].split()[:3]]
            for i in range(len(expanded_symbols))
        ],
        dtype=float,
    )
    if coord_mode.startswith(("d", "f")):
        coords = coords @ cell
    else:
        coords *= scale
    return expanded_symbols, coords


def read_structure(structure_file: str | Path) -> tuple[list[str], np.ndarray]:
    if Path(structure_file).suffix.lower() == ".xyz":
        return read_xyz_structure(structure_file)
    return read_vasp_structure(structure_file)


def molecular_mass_g_mol(symbols: list[str]) -> float:
    missing = sorted({symbol for symbol in symbols if symbol not in ATOMIC_MASSES})
    if missing:
        raise ValueError(f"Missing atomic masses for: {', '.join(missing)}")
    return float(sum(ATOMIC_MASSES[symbol] for symbol in symbols))


def save_xyz(
    path: str | Path,
    symbols: list[str],
    coords: np.ndarray,
    comment: str = "",
) -> Path:
    path = Path(path)
    lines = [str(len(symbols)), comment]
    lines.extend(
        f"{symbol} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}"
        for symbol, coord in zip(symbols, coords)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def structure_to_packmol_xyz(
    config: WorkflowInput,
    structure_file: str | Path,
    *,
    center_at: list[float] | None = None,
    label: str | None = None,
    output_stem: str | None = None,
) -> Path:
    source = _path(config.workdir, structure_file)
    if (
        source.suffix.lower() == ".xyz"
        and center_at is None
        and label is None
        and output_stem is None
    ):
        return source

    output_dir = _path(config.workdir, config.packmol.generated_structure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    output = output_dir / f"{output_stem or source.stem}{suffix}.xyz"

    symbols, coords = read_structure(source)
    if center_at is not None:
        coords = coords - coords.mean(axis=0) + np.asarray(center_at, dtype=float)
    return save_xyz(output, symbols, coords, f"Converted from {source.name}")


def prepare_structures(config: WorkflowInput) -> WorkflowInput:
    solute_1_path = _path(config.workdir, config.solute_1_file)
    if not solute_1_path.exists():
        raise FileNotFoundError(f"Missing solute_1 structure: {solute_1_path}")
    solute_1_symbols, _ = read_structure(solute_1_path)
    config.n_solute_1_atoms = len(solute_1_symbols)

    for system in config.systems:
        if system.n_solute_1_molecules < 1:
            raise ValueError(f"{system.name} must contain at least one solute_1 molecule.")
        total_fraction = sum(s.volume_fraction for s in system.solvents)
        if not np.isclose(total_fraction, 1.0):
            raise ValueError(f"Solvent volume fractions for {system.name} must sum to 1.0.")
        for solvent in system.solvents:
            solvent_path = _path(config.workdir, solvent.structure_file)
            if not solvent_path.exists():
                raise FileNotFoundError(
                    f"Missing {solvent.role} structure ({solvent.chemical_name}): {solvent_path}"
                )
            symbols, _ = read_structure(solvent_path)
            solvent.n_atoms = len(symbols)
            solvent.molar_mass_g_mol = molecular_mass_g_mol(symbols)
    return config


def estimate_system_atom_count(config: WorkflowInput, system: SystemInput) -> int:
    if config.n_solute_1_atoms is None:
        prepare_structures(config)
    total_atoms = config.n_solute_1_atoms * system.n_solute_1_molecules
    for solvent in system.solvents:
        if solvent.n_atoms is None:
            raise ValueError(f"Missing atom count for {solvent.role}.")
        total_atoms += solvent.n_atoms * solvent.n_molecules
    return int(total_atoms)


def estimate_neighbor_count(config: WorkflowInput, system: SystemInput) -> int:
    n_atoms = estimate_system_atom_count(config, system)
    volume_a3 = system.box_length_a**3
    return int(round(PFP_NEIGHBOR_DENSITY_FACTOR * n_atoms * (n_atoms / volume_a3)))


def actual_volume_fraction(solvent: SolventInput, system: SystemInput) -> float:
    volumes = [
        c.n_molecules * c.molar_mass_g_mol / c.density_g_cm3
        for c in system.solvents
    ]
    index = system.solvents.index(solvent)
    return float(volumes[index] / sum(volumes))


def validate_sizes(
    config: WorkflowInput,
    max_atoms: int | None,
    max_neighbors: int | None,
) -> None:
    prepare_structures(config)
    failures = []
    for system in config.systems:
        atoms = estimate_system_atom_count(config, system)
        neighbors = estimate_neighbor_count(config, system)
        if (max_atoms is not None and atoms > max_atoms) or (
            max_neighbors is not None and neighbors > max_neighbors
        ):
            failures.append(
                f"{system.name}: atoms={atoms:,}, estimated_neighbors={neighbors:,}"
            )
    if failures:
        raise ValueError("System size limit exceeded:\n" + "\n".join(failures))


def _packmol_path(config: WorkflowInput, system: SystemInput, template: str) -> Path:
    return _path(config.workdir, config.packmol.output_dir) / template.format(
        name=_safe_name(system.name)
    )


def _packmol_ref(config: WorkflowInput, path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(config.workdir).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_packmol_input(config: WorkflowInput, system: SystemInput) -> Path:
    packmol = config.packmol
    box_length = system.box_length_a
    if packmol.margin <= 0 or packmol.margin * 2 >= box_length:
        raise ValueError("Packmol margin must be positive and below half the box length.")

    output_dir = _path(config.workdir, packmol.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    center = [box_length / 2.0] * 3
    fix_single_solute_1 = (
        packmol.fixed_single_solute_1 and system.n_solute_1_molecules == 1
    )
    solute_1_xyz = structure_to_packmol_xyz(
        config,
        config.solute_1_file,
        center_at=center if fix_single_solute_1 else None,
        label="fixed" if fix_single_solute_1 else None,
        output_stem=f"{_safe_name(system.name)}_solute_1",
    )
    output_xyz = _packmol_path(config, system, packmol.output_xyz_template)

    if fix_single_solute_1:
        solute_1_block = f"""structure {_packmol_ref(config, solute_1_xyz)}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure"""
    else:
        solute_1_block = f"""structure {_packmol_ref(config, solute_1_xyz)}
  number {system.n_solute_1_molecules}
  inside box {packmol.margin} {packmol.margin} {packmol.margin} {box_length - packmol.margin} {box_length - packmol.margin} {box_length - packmol.margin}
end structure"""

    solvent_blocks = []
    for solvent in system.solvents:
        solvent_xyz = structure_to_packmol_xyz(
            config,
            solvent.structure_file,
            output_stem=f"{_safe_name(system.name)}_{solvent.role}",
        )
        solvent_blocks.append(
            f"""structure {_packmol_ref(config, solvent_xyz)}
  number {solvent.n_molecules}
  inside box {packmol.margin} {packmol.margin} {packmol.margin} {box_length - packmol.margin} {box_length - packmol.margin} {box_length - packmol.margin}
end structure"""
        )

    content = "\n\n".join(
        [
            f"tolerance {packmol.tolerance}\n"
            f"filetype xyz\n"
            f"output {_packmol_ref(config, output_xyz)}",
            solute_1_block,
            *solvent_blocks,
        ]
    )
    input_path = _packmol_path(config, system, packmol.packmol_input_template)
    input_path.write_text(content + "\n", encoding="utf-8")
    return input_path


def write_packmol_inputs(config: WorkflowInput) -> list[Path]:
    prepare_structures(config)
    return [write_packmol_input(config, system) for system in config.systems]


def resolve_packmol_executable(executable: str) -> str:
    """Resolve Packmol from config, an environment variable, PATH, or Python env."""
    requested = os.environ.get("PACKMOL_EXECUTABLE", executable)
    requested_path = Path(requested).expanduser()
    if requested_path.is_file():
        return str(requested_path.resolve())

    resolved = shutil.which(requested)
    if resolved is not None:
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

    checked = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        f"Packmol executable was not found for {requested!r}.\n"
        f"Python executable: {sys.executable}\n"
        f"Also checked:\n{checked}\n"
        "Run this script with the Python environment where Packmol is installed, "
        "or set CONFIG.runtime.packmol_executable / PACKMOL_EXECUTABLE."
    )


def validate_packed_output(
    config: WorkflowInput,
    system: SystemInput,
    output_xyz: str | Path,
) -> None:
    """Confirm that Packmol created an XYZ with the expected atom count."""
    output_xyz = Path(output_xyz)
    if not output_xyz.is_file():
        raise FileNotFoundError(
            f"Packmol finished without creating the expected output: {output_xyz}"
        )
    first_line = output_xyz.read_text(encoding="utf-8").splitlines()[0].strip()
    actual_atoms = int(first_line)
    expected_atoms = estimate_system_atom_count(config, system)
    if actual_atoms != expected_atoms:
        raise ValueError(
            f"Packed XYZ atom-count mismatch for {system.name}: "
            f"expected {expected_atoms}, found {actual_atoms}."
        )


def run_packmol(
    config: WorkflowInput,
    executable: str = "packmol",
    *,
    skip_existing: bool = False,
    quiet: bool = True,
) -> list[Path]:
    resolved = resolve_packmol_executable(executable)
    print(f"Resolved Packmol executable: {resolved}")

    outputs = []
    for system in config.systems:
        input_path = write_packmol_input(config, system)
        output_xyz = _packmol_path(config, system, config.packmol.output_xyz_template)
        if skip_existing and output_xyz.exists():
            validate_packed_output(config, system, output_xyz)
            print(f"Skipping existing Packmol output: {output_xyz}")
            outputs.append(output_xyz)
            continue

        log_path = input_path.with_suffix(".log")
        print(f"Running Packmol for {system.name}; log: {log_path}")
        with input_path.open("rb") as stdin:
            if quiet:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [resolved],
                        cwd=config.workdir,
                        stdin=stdin,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            else:
                subprocess.run(
                    [resolved], cwd=config.workdir, stdin=stdin, check=True
                )
        validate_packed_output(config, system, output_xyz)
        print(
            f"Packmol finished and verified: {output_xyz} "
            f"({estimate_system_atom_count(config, system):,} atoms)"
        )
        outputs.append(output_xyz)
    return outputs


def main(config: WorkflowInput) -> None:
    validate_sizes(
        config,
        config.runtime.max_atoms,
        config.runtime.max_estimated_neighbors,
    )

    config_path = _path(config.workdir, "solute_solvent_packmol_config.json")
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    input_paths = write_packmol_inputs(config)

    print("Solute/solvent system settings:")
    for system in config.systems:
        counts = ", ".join(
            f"{s.role} [{s.chemical_name}]={s.n_molecules} "
            f"(actual_v={100 * actual_volume_fraction(s, system):.3f}%)"
            for s in system.solvents
        )
        print(
            f"{system.name:58s} box={system.box_length_a:g} A  "
            f"solute_1={system.n_solute_1_molecules}  {counts}  "
            f"atoms={estimate_system_atom_count(config, system):,}  "
            f"estimated_neighbors={estimate_neighbor_count(config, system):,}"
        )

    print("\nGenerated Packmol inputs:")
    for path in input_paths:
        print(" ", path)

    if config.runtime.run_packmol:
        outputs = run_packmol(
            config,
            executable=config.runtime.packmol_executable,
            skip_existing=config.runtime.skip_existing,
            quiet=config.runtime.quiet,
        )
        print("\nGenerated packed structures:")
        for path in outputs:
            print(" ", path)


if __name__ == "__main__":
    main(CONFIG)
