"""Build pure-solvent, binary, and ternary OPA systems with Packmol only."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
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
class ComponentInput:
    name: str
    structure_file: str
    density_g_cm3: float
    volume_fraction: float
    n_molecules: int
    molar_mass_g_mol: float | None = None
    n_atoms: int | None = None


@dataclass
class MixtureInput:
    name: str
    box_length_a: float
    components: list[ComponentInput]
    n_opa_molecules: int = 1


@dataclass
class PackmolInput:
    margin: float = 2.0
    tolerance: float = 2.2
    fixed_opa: bool = True
    generated_structure_dir: str = "packmol_structures"
    output_dir: str = "packmol_systems"
    packmol_input_template: str = "{name}_packmol.inp"
    output_xyz_template: str = "{name}_opa_box.xyz"


@dataclass
class RuntimeInput:
    packmol_executable: str = "packmol"
    run_packmol: bool = True
    skip_existing: bool = False
    quiet: bool = True
    max_atoms: int | None = 30_000
    max_estimated_neighbors: int | None = 1_650_000


def _component(
    name: str,
    structure_file: str,
    density: float,
    volume_fraction: float,
    n_molecules: int,
) -> ComponentInput:
    return ComponentInput(
        name=name,
        structure_file=structure_file,
        density_g_cm3=density,
        volume_fraction=volume_fraction,
        n_molecules=n_molecules,
    )


@dataclass
class WorkflowInput:
    workdir: str = "."
    opa_file: str = "opa.vasp"
    packmol: PackmolInput = field(default_factory=PackmolInput)
    runtime: RuntimeInput = field(default_factory=RuntimeInput)
    mixtures: list[MixtureInput] = field(default_factory=list)
    n_opa_atoms: int | None = None


# =========================
# Editable configuration
# =========================
CONFIG = WorkflowInput(
    workdir=".",
    opa_file="opa.vasp",
    packmol=PackmolInput(
        margin=2.0,
        tolerance=2.2,
        fixed_opa=True,
        generated_structure_dir="packmol_structures",
        output_dir="packmol_systems",
        packmol_input_template="{name}_packmol.inp",
        output_xyz_template="{name}_opa_box.xyz",
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
    mixtures=[
        MixtureInput(
            name="example_pure_n-heptane",
            box_length_a=47.0,
            n_opa_molecules=1,
            components=[
                _component("n-heptane", "n-heptane.vasp", 0.684, 1.00, 427),
            ],
        ),
        MixtureInput(
            name="example_binary_acetone_10_n-heptane_90",
            box_length_a=48.0,
            n_opa_molecules=1,
            components=[
                _component("acetone", "Actone.vasp", 0.7845, 0.10, 90),
                _component("n-heptane", "n-heptane.vasp", 0.684, 0.90, 409),
            ],
        ),
        MixtureInput(
            name="example_ternary_acetone_10_n-heptane_80_thf_10",
            box_length_a=47.0,
            n_opa_molecules=2,
            components=[
                _component("acetone", "Actone.vasp", 0.7845, 0.10, 84),
                _component("n-heptane", "n-heptane.vasp", 0.684, 0.80, 341),
                _component("thf", "thf.vasp", 0.889, 0.10, 77),
            ],
        ),
    ],
)


def _path(workdir: str | Path, filename: str | Path) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else Path(workdir) / path


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


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
) -> Path:
    source = _path(config.workdir, structure_file)
    if source.suffix.lower() == ".xyz" and center_at is None:
        return source

    output_dir = _path(config.workdir, config.packmol.generated_structure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    output = output_dir / f"{source.stem}{suffix}.xyz"

    symbols, coords = read_structure(source)
    if center_at is not None:
        coords = coords - coords.mean(axis=0) + np.asarray(center_at, dtype=float)
    return save_xyz(output, symbols, coords, f"Converted from {source.name}")


def prepare_structures(config: WorkflowInput) -> WorkflowInput:
    opa_path = _path(config.workdir, config.opa_file)
    if not opa_path.exists():
        raise FileNotFoundError(f"Missing OPA structure: {opa_path}")
    opa_symbols, _ = read_structure(opa_path)
    config.n_opa_atoms = len(opa_symbols)

    for mixture in config.mixtures:
        if mixture.n_opa_molecules < 1:
            raise ValueError(f"{mixture.name} must contain at least one OPA molecule.")
        total_fraction = sum(c.volume_fraction for c in mixture.components)
        if not np.isclose(total_fraction, 1.0):
            raise ValueError(f"Volume fractions for {mixture.name} must sum to 1.0.")
        for component in mixture.components:
            component_path = _path(config.workdir, component.structure_file)
            if not component_path.exists():
                raise FileNotFoundError(f"Missing component structure: {component_path}")
            symbols, _ = read_structure(component_path)
            component.n_atoms = len(symbols)
            component.molar_mass_g_mol = molecular_mass_g_mol(symbols)
    return config


def estimate_system_atom_count(config: WorkflowInput, mixture: MixtureInput) -> int:
    if config.n_opa_atoms is None:
        prepare_structures(config)
    total_atoms = config.n_opa_atoms * mixture.n_opa_molecules
    for component in mixture.components:
        if component.n_atoms is None:
            raise ValueError(f"Missing atom count for {component.name}.")
        total_atoms += component.n_atoms * component.n_molecules
    return int(total_atoms)


def estimate_neighbor_count(config: WorkflowInput, mixture: MixtureInput) -> int:
    n_atoms = estimate_system_atom_count(config, mixture)
    volume_a3 = mixture.box_length_a**3
    return int(round(PFP_NEIGHBOR_DENSITY_FACTOR * n_atoms * (n_atoms / volume_a3)))


def actual_volume_fraction(component: ComponentInput, mixture: MixtureInput) -> float:
    volumes = [
        c.n_molecules * c.molar_mass_g_mol / c.density_g_cm3
        for c in mixture.components
    ]
    index = mixture.components.index(component)
    return float(volumes[index] / sum(volumes))


def validate_sizes(
    config: WorkflowInput,
    max_atoms: int | None,
    max_neighbors: int | None,
) -> None:
    prepare_structures(config)
    failures = []
    for mixture in config.mixtures:
        atoms = estimate_system_atom_count(config, mixture)
        neighbors = estimate_neighbor_count(config, mixture)
        if (max_atoms is not None and atoms > max_atoms) or (
            max_neighbors is not None and neighbors > max_neighbors
        ):
            failures.append(
                f"{mixture.name}: atoms={atoms:,}, estimated_neighbors={neighbors:,}"
            )
    if failures:
        raise ValueError("System size limit exceeded:\n" + "\n".join(failures))


def _packmol_path(config: WorkflowInput, mixture: MixtureInput, template: str) -> Path:
    return _path(config.workdir, config.packmol.output_dir) / template.format(
        name=_safe_name(mixture.name)
    )


def _packmol_ref(config: WorkflowInput, path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(config.workdir).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_packmol_input(config: WorkflowInput, mixture: MixtureInput) -> Path:
    packmol = config.packmol
    box_length = mixture.box_length_a
    if packmol.margin <= 0 or packmol.margin * 2 >= box_length:
        raise ValueError("Packmol margin must be positive and below half the box length.")

    output_dir = _path(config.workdir, packmol.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    center = [box_length / 2.0] * 3
    fix_single_opa = packmol.fixed_opa and mixture.n_opa_molecules == 1
    opa_xyz = structure_to_packmol_xyz(
        config,
        config.opa_file,
        center_at=center if fix_single_opa else None,
        label=f"{_safe_name(mixture.name)}_fixed" if fix_single_opa else None,
    )
    output_xyz = _packmol_path(config, mixture, packmol.output_xyz_template)

    if fix_single_opa:
        opa_block = f"""structure {_packmol_ref(config, opa_xyz)}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure"""
    else:
        opa_block = f"""structure {_packmol_ref(config, opa_xyz)}
  number {mixture.n_opa_molecules}
  inside box {packmol.margin} {packmol.margin} {packmol.margin} {box_length - packmol.margin} {box_length - packmol.margin} {box_length - packmol.margin}
end structure"""

    component_blocks = []
    for component in mixture.components:
        component_xyz = structure_to_packmol_xyz(config, component.structure_file)
        component_blocks.append(
            f"""structure {_packmol_ref(config, component_xyz)}
  number {component.n_molecules}
  inside box {packmol.margin} {packmol.margin} {packmol.margin} {box_length - packmol.margin} {box_length - packmol.margin} {box_length - packmol.margin}
end structure"""
        )

    content = "\n\n".join(
        [
            f"tolerance {packmol.tolerance}\n"
            f"filetype xyz\n"
            f"output {_packmol_ref(config, output_xyz)}",
            opa_block,
            *component_blocks,
        ]
    )
    input_path = _packmol_path(config, mixture, packmol.packmol_input_template)
    input_path.write_text(content + "\n", encoding="utf-8")
    return input_path


def write_packmol_inputs(config: WorkflowInput) -> list[Path]:
    prepare_structures(config)
    return [write_packmol_input(config, mixture) for mixture in config.mixtures]


def run_packmol(
    config: WorkflowInput,
    executable: str = "packmol",
    *,
    skip_existing: bool = False,
    quiet: bool = True,
) -> list[Path]:
    resolved = executable if Path(executable).exists() else shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(
            f"Packmol executable was not found: {executable!r}. "
            "Set PACKMOL_EXECUTABLE to its absolute path."
        )

    outputs = []
    for mixture in config.mixtures:
        input_path = write_packmol_input(config, mixture)
        output_xyz = _packmol_path(config, mixture, config.packmol.output_xyz_template)
        if skip_existing and output_xyz.exists():
            print(f"Skipping existing Packmol output: {output_xyz}")
            outputs.append(output_xyz)
            continue

        log_path = input_path.with_suffix(".log")
        print(f"Running Packmol for {mixture.name}; log: {log_path}")
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
        print(f"Packmol finished: {output_xyz}")
        outputs.append(output_xyz)
    return outputs


def main(config: WorkflowInput) -> None:
    validate_sizes(
        config,
        config.runtime.max_atoms,
        config.runtime.max_estimated_neighbors,
    )

    config_path = _path(config.workdir, "packmol_config.json")
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    input_paths = write_packmol_inputs(config)

    print("Mixture settings:")
    for mixture in config.mixtures:
        counts = ", ".join(
            f"{c.name}={c.n_molecules} "
            f"(actual_v={100 * actual_volume_fraction(c, mixture):.3f}%)"
            for c in mixture.components
        )
        print(
            f"{mixture.name:50s} box={mixture.box_length_a:g} A  "
            f"OPA={mixture.n_opa_molecules}  {counts}  "
            f"atoms={estimate_system_atom_count(config, mixture):,}  "
            f"estimated_neighbors={estimate_neighbor_count(config, mixture):,}"
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
