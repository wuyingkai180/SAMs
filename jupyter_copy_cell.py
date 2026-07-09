# Integrated Jupyter workflow.
# Copy this whole file into one Jupyter cell.
# Keep the notebook in the same directory as:
#   opa.vasp, Actone.vasp, n-heptane.vasp, prol.vasp, thf.vasp, Toluene.vasp

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


AVOGADRO = 6.02214076e23
# Calibrated from the Matlantis/PFP error for the 60 A acetone box:
# 17,631 atoms produced 2,396,685 input neighbors. The estimate scales with
# atom count and atom number density, which is the relevant risk for this API
# limit. Keep the default guard below the server-side max_neighbors=1,700,001.
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
    name: str
    structure_file: str
    density_g_cm3: float
    box_length_a: float | None = None
    molar_mass_g_mol: float | None = None
    n_atoms: int | None = None
    n_molecules: int | None = None


@dataclass
class StructureInput:
    opa_file: str = "opa.vasp"
    solvents: list[SolventInput] = field(
        default_factory=lambda: [
            SolventInput("acetone", "Actone.vasp", 0.7845),
            SolventInput("n-heptane", "n-heptane.vasp", 0.684),
            SolventInput("prol", "prol.vasp", 1.35),
            SolventInput("thf", "thf.vasp", 0.889),
            SolventInput("toluene", "Toluene.vasp", 0.867),
        ]
    )
    n_opa_atoms: int | None = None


@dataclass
class PackmolInput:
    box_length: float = 100.0
    margin: float = 2.0
    tolerance: float = 2.2
    fixed_opa: bool = True
    opa_center: list[float] | None = None
    generated_structure_dir: str = "packmol_structures"
    output_dir: str = "packmol_systems"
    packmol_input_template: str = "{name}_packmol.inp"
    output_xyz_template: str = "{name}_opa_box.xyz"


@dataclass
class MDInput:
    input_xyz: str = "packmol_systems/acetone_opa_box.xyz"
    output_traj: str = "md_300K.traj"
    output_xyz: str = "final_300K.xyz"
    temperature: float = 300.0
    timestep_fs: float = 1.0
    friction_per_fs: float = 0.01
    n_steps: int = 100000
    log_interval: int = 100
    random_seed: int | None = 20260707


@dataclass
class AnalysisInput:
    traj_file: str = "md_300K.traj"
    dt_fs: float = 1.0
    save_interval: int = 100
    output_csv: str = "opa_motion_force.csv"
    displacement_png: str = "opa_displacement.png"
    force_png: str = "opa_force.png"


@dataclass
class WorkflowInput:
    workdir: str = "."
    structure: StructureInput = field(default_factory=StructureInput)
    packmol: PackmolInput = field(default_factory=PackmolInput)
    md: MDInput = field(default_factory=MDInput)
    analysis: AnalysisInput = field(default_factory=AnalysisInput)


def _path(workdir: str | Path, filename: str | Path) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else Path(workdir) / path


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def read_xyz_atom_count(xyz_file: str | Path) -> int:
    first_line = Path(xyz_file).read_text(encoding="utf-8").splitlines()[0].strip()
    return int(first_line)


def read_xyz_structure(xyz_file: str | Path) -> tuple[list[str], np.ndarray]:
    lines = Path(xyz_file).read_text(encoding="utf-8").splitlines()
    n_atoms = read_xyz_atom_count(xyz_file)
    symbols = []
    coords = []
    for line in lines[2 : 2 + n_atoms]:
        fields = line.split()
        symbols.append(fields[0])
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return symbols, np.array(coords)


def read_vasp_structure(vasp_file: str | Path) -> tuple[list[str], np.ndarray]:
    lines = Path(vasp_file).read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError(f"{vasp_file} does not look like a valid POSCAR/VASP file.")

    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])
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
        ]
    )
    if coord_mode.startswith(("d", "f")):
        coords = coords @ cell
    else:
        coords *= scale
    return expanded_symbols, coords


def read_structure_symbols_coords(structure_file: str | Path) -> tuple[list[str], np.ndarray]:
    if Path(structure_file).suffix.lower() == ".xyz":
        return read_xyz_structure(structure_file)
    return read_vasp_structure(structure_file)


def molecular_mass_g_mol(structure_file: str | Path) -> float:
    symbols, _coords = read_structure_symbols_coords(structure_file)
    missing = sorted({symbol for symbol in symbols if symbol not in ATOMIC_MASSES})
    if missing:
        raise ValueError(f"Missing atomic masses for: {', '.join(missing)}")
    return float(sum(ATOMIC_MASSES[symbol] for symbol in symbols))


def estimate_molecule_count(
    box_length_a: float,
    density_g_cm3: float,
    molar_mass_g_mol: float,
) -> int:
    volume_cm3 = box_length_a**3 * 1e-24
    mass_g = density_g_cm3 * volume_cm3
    n_mol = mass_g / molar_mass_g_mol
    return int(round(n_mol * AVOGADRO))


def get_solvent_box_length(config: WorkflowInput, solvent: SolventInput) -> float:
    return solvent.box_length_a or config.packmol.box_length


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
    src = _path(config.workdir, structure_file)
    if src.suffix.lower() == ".xyz" and center_at is None:
        return src

    out_dir = _path(config.workdir, config.packmol.generated_structure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    out = out_dir / f"{src.stem}{suffix}.xyz"

    symbols, coords = read_structure_symbols_coords(src)
    if center_at is not None:
        coords = coords - coords.mean(axis=0) + np.array(center_at, dtype=float)
    return save_xyz(out, symbols, coords, f"Converted from {src.name}")


def prepare_structure_input(config: WorkflowInput) -> WorkflowInput:
    opa_path = _path(config.workdir, config.structure.opa_file)
    if not opa_path.exists():
        raise FileNotFoundError(f"Missing OPA structure: {opa_path}")

    opa_symbols, _opa_coords = read_structure_symbols_coords(opa_path)
    config.structure.n_opa_atoms = len(opa_symbols)

    for solvent in config.structure.solvents:
        solvent_path = _path(config.workdir, solvent.structure_file)
        if not solvent_path.exists():
            raise FileNotFoundError(f"Missing solvent structure: {solvent_path}")
        symbols, _coords = read_structure_symbols_coords(solvent_path)
        solvent.n_atoms = len(symbols)
        solvent.molar_mass_g_mol = molecular_mass_g_mol(solvent_path)
        solvent.n_molecules = estimate_molecule_count(
            get_solvent_box_length(config, solvent),
            solvent.density_g_cm3,
            solvent.molar_mass_g_mol,
        )
    return config


def estimate_system_atom_count(config: WorkflowInput, solvent: SolventInput) -> int:
    if config.structure.n_opa_atoms is None:
        prepare_structure_input(config)
    if solvent.n_atoms is None or solvent.n_molecules is None:
        raise ValueError(f"Solvent metadata is incomplete for {solvent.name}.")
    return int(config.structure.n_opa_atoms + solvent.n_atoms * solvent.n_molecules)


def estimate_pfp_neighbor_count(config: WorkflowInput, solvent: SolventInput) -> int:
    n_atoms = estimate_system_atom_count(config, solvent)
    volume_a3 = get_solvent_box_length(config, solvent) ** 3
    atom_density_a3 = n_atoms / volume_a3
    return int(round(PFP_NEIGHBOR_DENSITY_FACTOR * n_atoms * atom_density_a3))


def validate_md_system_sizes(
    config: WorkflowInput,
    max_atoms: int | None,
    system_names: list[str] | None = None,
) -> None:
    if max_atoms is None:
        return

    prepare_structure_input(config)
    selected = (
        {_safe_name(name).lower() for name in system_names}
        if system_names is not None
        else None
    )
    too_large = []
    for solvent in config.structure.solvents:
        if selected is not None and _safe_name(solvent.name).lower() not in selected:
            continue
        n_atoms = estimate_system_atom_count(config, solvent)
        if n_atoms > max_atoms:
            too_large.append((solvent.name, n_atoms, solvent.n_molecules))

    if too_large:
        lines = [
            f"{name}: {n_atoms} atoms ({n_molecules} solvent molecules)"
            for name, n_atoms, n_molecules in too_large
        ]
        raise ValueError(
            "Estimated MD system exceeds MD_MAX_ATOMS before calling Matlantis/PFP.\n"
            + "\n".join(lines)
            + "\nReduce BOX_LENGTH_A or increase MD_MAX_ATOMS if your API plan supports it."
        )


def validate_mlip_neighbor_limit(
    config: WorkflowInput,
    max_neighbors: int | None,
    system_names: list[str] | None = None,
) -> None:
    if max_neighbors is None:
        return

    prepare_structure_input(config)
    selected = (
        {_safe_name(name).lower() for name in system_names}
        if system_names is not None
        else None
    )
    too_large = []
    for solvent in config.structure.solvents:
        if selected is not None and _safe_name(solvent.name).lower() not in selected:
            continue
        n_atoms = estimate_system_atom_count(config, solvent)
        neighbors = estimate_pfp_neighbor_count(config, solvent)
        if neighbors > max_neighbors:
            too_large.append((solvent.name, n_atoms, solvent.n_molecules, neighbors))

    if too_large:
        lines = [
            (
                f"{name}: estimated_neighbors={neighbors:,}, atoms={n_atoms:,}, "
                f"solvent_molecules={n_molecules}"
            )
            for name, n_atoms, n_molecules, neighbors in too_large
        ]
        raise ValueError(
            "Estimated MLIP neighbor count exceeds MD_MAX_ESTIMATED_NEIGHBORS before "
            "calling Matlantis/PFP.\n"
            + "\n".join(lines)
            + "\nReduce BOX_LENGTH_A or set SYSTEM_NAMES to a smaller test subset."
        )


def _packmol_path(config: WorkflowInput, solvent: SolventInput, template: str) -> Path:
    return _path(config.workdir, config.packmol.output_dir) / template.format(
        name=_safe_name(solvent.name)
    )


def _packmol_ref(config: WorkflowInput, path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(config.workdir).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def write_packmol_input_for_solvent(config: WorkflowInput, solvent: SolventInput) -> Path:
    p = config.packmol
    box_length = get_solvent_box_length(config, solvent)
    if p.margin <= 0 or p.margin * 2 >= box_length:
        raise ValueError("Packmol margin must be positive and smaller than half the box length.")

    out_dir = _path(config.workdir, p.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opa_center = p.opa_center or [box_length / 2.0] * 3
    opa_xyz = structure_to_packmol_xyz(
        config,
        config.structure.opa_file,
        center_at=opa_center if p.fixed_opa else None,
        label="fixed" if p.fixed_opa else None,
    )
    solvent_xyz = structure_to_packmol_xyz(config, solvent.structure_file)
    output_xyz = _packmol_path(config, solvent, p.output_xyz_template)

    if p.fixed_opa:
        opa_block = f"""
structure {_packmol_ref(config, opa_xyz)}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure
""".strip()
    else:
        opa_block = f"""
structure {_packmol_ref(config, opa_xyz)}
  number 1
  inside box {p.margin} {p.margin} {p.margin} {box_length - p.margin} {box_length - p.margin} {box_length - p.margin}
end structure
""".strip()

    content = f"""
tolerance {p.tolerance}
filetype xyz
output {_packmol_ref(config, output_xyz)}

{opa_block}

structure {_packmol_ref(config, solvent_xyz)}
  number {solvent.n_molecules}
  inside box {p.margin} {p.margin} {p.margin} {box_length - p.margin} {box_length - p.margin} {box_length - p.margin}
end structure
""".strip()

    path = _packmol_path(config, solvent, p.packmol_input_template)
    path.write_text(content + "\n", encoding="utf-8")
    return path


def write_packmol_inputs(config: WorkflowInput) -> list[Path]:
    prepare_structure_input(config)
    return [
        write_packmol_input_for_solvent(config, solvent)
        for solvent in config.structure.solvents
    ]


def run_packmol(
    config: WorkflowInput,
    executable: str = "packmol",
    *,
    skip_existing: bool = True,
    quiet: bool = True,
) -> list[Path]:
    resolved_executable = shutil.which(executable) if not Path(executable).exists() else executable
    if resolved_executable is None:
        raise FileNotFoundError(
            "Packmol executable was not found. Install/load Packmol in this "
            "environment, or set PACKMOL_EXECUTABLE to the absolute path of "
            "the packmol binary, for example '/path/to/packmol'."
        )

    prepare_structure_input(config)
    outputs = []
    for solvent in config.structure.solvents:
        inp = write_packmol_input_for_solvent(config, solvent)
        output_xyz = _packmol_path(config, solvent, config.packmol.output_xyz_template)
        if skip_existing and output_xyz.exists():
            print(f"Skipping existing Packmol output: {output_xyz}")
            outputs.append(output_xyz)
            continue
        log_path = inp.with_suffix(".log")
        print(f"Running Packmol for {solvent.name}; log: {log_path}")
        with inp.open("rb") as stdin:
            if quiet:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [resolved_executable],
                        cwd=config.workdir,
                        stdin=stdin,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            else:
                subprocess.run([resolved_executable], cwd=config.workdir, stdin=stdin, check=True)
        print(f"Packmol finished for {solvent.name}: {output_xyz}")
        outputs.append(output_xyz)
    return outputs


def save_config(config: WorkflowInput, json_file: str | Path = "workflow_input.json") -> Path:
    path = Path(json_file)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return path


def load_md_atoms(config: WorkflowInput):
    from ase.io import read

    atoms = read(_path(config.workdir, config.md.input_xyz))
    atoms.set_cell([config.packmol.box_length] * 3)
    atoms.set_pbc([True, True, True])
    atoms.info["opa_indices"] = list(range(config.structure.n_opa_atoms))
    atoms.info["solvent_indices"] = list(range(config.structure.n_opa_atoms, len(atoms)))
    return atoms


def run_mlip_md(config: WorkflowInput, calc: Any):
    from ase import units
    from ase.io import write
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    atoms = load_md_atoms(config)
    atoms.calc = calc
    opa_indices = list(range(config.structure.n_opa_atoms))

    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=config.md.temperature,
        rng=np.random.default_rng(config.md.random_seed),
    )
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = Langevin(
        atoms,
        config.md.timestep_fs * units.fs,
        temperature_K=config.md.temperature,
        friction=config.md.friction_per_fs / units.fs,
    )

    traj_path = _path(config.workdir, config.md.output_traj)
    traj = Trajectory(traj_path, "w", atoms)

    def print_status() -> None:
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = atoms.get_temperature()
        forces = atoms.get_forces()
        f_opa_norm = np.linalg.norm(forces[opa_indices].sum(axis=0))
        print(
            f"Step {dyn.get_number_of_steps():8d} | "
            f"T = {temp:8.2f} K | "
            f"Epot = {epot:14.6f} eV | "
            f"Etot = {epot + ekin:14.6f} eV | "
            f"|F_OPA_total| = {f_opa_norm:12.6f} eV/A"
        )

    dyn.attach(traj.write, interval=config.md.log_interval)
    dyn.attach(print_status, interval=config.md.log_interval)

    dyn.run(config.md.n_steps)
    traj.close()

    final_xyz = _path(config.workdir, config.md.output_xyz)
    write(final_xyz, atoms)
    print(f"Trajectory saved to {traj_path}")
    print(f"Final structure saved to {final_xyz}")
    return atoms


def analyze_opa_motion(config: WorkflowInput):
    import pandas as pd
    from ase.io import read

    frames = read(_path(config.workdir, config.analysis.traj_file), index=":")
    opa_indices = list(range(config.structure.n_opa_atoms))

    times_ps = []
    opa_com_list = []
    opa_force_list = []
    opa_force_norm_list = []

    for i, atoms in enumerate(frames):
        time_ps = i * config.analysis.save_interval * config.analysis.dt_fs / 1000.0
        opa = atoms[opa_indices]
        forces = atoms.get_forces()
        f_opa = forces[opa_indices].sum(axis=0)

        times_ps.append(time_ps)
        opa_com_list.append(opa.get_center_of_mass())
        opa_force_list.append(f_opa)
        opa_force_norm_list.append(np.linalg.norm(f_opa))

    opa_com_arr = np.array(opa_com_list)
    opa_force_arr = np.array(opa_force_list)
    disp = opa_com_arr - opa_com_arr[0]
    disp_norm = np.linalg.norm(disp, axis=1)

    df = pd.DataFrame(
        {
            "time_ps": times_ps,
            "OPA_COM_x_A": opa_com_arr[:, 0],
            "OPA_COM_y_A": opa_com_arr[:, 1],
            "OPA_COM_z_A": opa_com_arr[:, 2],
            "OPA_disp_x_A": disp[:, 0],
            "OPA_disp_y_A": disp[:, 1],
            "OPA_disp_z_A": disp[:, 2],
            "OPA_disp_norm_A": disp_norm,
            "F_OPA_x_eV_A": opa_force_arr[:, 0],
            "F_OPA_y_eV_A": opa_force_arr[:, 1],
            "F_OPA_z_eV_A": opa_force_arr[:, 2],
            "F_OPA_norm_eV_A": opa_force_norm_list,
        }
    )
    df.to_csv(_path(config.workdir, config.analysis.output_csv), index=False)
    return df


def plot_opa_motion(config: WorkflowInput, df=None) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    if df is None:
        df = pd.read_csv(_path(config.workdir, config.analysis.output_csv))

    plt.figure()
    plt.plot(df["time_ps"], df["OPA_disp_norm_A"])
    plt.xlabel("Time / ps")
    plt.ylabel("OPA COM displacement / A")
    plt.tight_layout()
    plt.savefig(_path(config.workdir, config.analysis.displacement_png), dpi=300)

    plt.figure()
    plt.plot(df["time_ps"], df["F_OPA_norm_eV_A"])
    plt.xlabel("Time / ps")
    plt.ylabel("|F_OPA| / eV A$^{-1}$")
    plt.tight_layout()
    plt.savefig(_path(config.workdir, config.analysis.force_png), dpi=300)


def _resolve_estimator_calc_mode(calc_mode: str | Any):
    from pfp_api_client.pfp.estimator import EstimatorCalcMode

    if isinstance(calc_mode, str):
        return getattr(EstimatorCalcMode, calc_mode)
    return calc_mode


def create_pfp_calculator(calc_mode: str | Any = "PBE_U_PLUS_D3"):
    """Create a Matlantis/PFP ASE calculator through pfp_api_client."""
    from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
    from pfp_api_client.pfp.estimator import Estimator

    estimator = Estimator(calc_mode=_resolve_estimator_calc_mode(calc_mode))
    return ASECalculator(estimator)


def create_matlantis_calculator(
    model_version: str = "v9.0.0",
    calc_mode: str | Any = "R2SCAN",
):
    """Create a Matlantis/PFP ASE calculator for a specific model version."""
    from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
    from pfp_api_client.pfp.estimator import Estimator

    estimator = Estimator(
        model_version=model_version,
        calc_mode=_resolve_estimator_calc_mode(calc_mode),
    )
    return ASECalculator(estimator)


def configure_system_outputs(
    config: WorkflowInput,
    solvent: SolventInput,
    results_dir: str | Path = "results",
) -> WorkflowInput:
    """Return a config copy with per-system input and output paths."""
    c = deepcopy(config)
    c.packmol.box_length = get_solvent_box_length(config, solvent)
    name = _safe_name(solvent.name)
    system_dir = _path(c.workdir, results_dir) / name
    system_dir.mkdir(parents=True, exist_ok=True)

    c.md.input_xyz = str(_packmol_path(c, solvent, c.packmol.output_xyz_template))
    c.md.output_traj = str(system_dir / f"{name}_md_300K.traj")
    c.md.output_xyz = str(system_dir / f"{name}_final_300K.xyz")

    c.analysis.traj_file = c.md.output_traj
    c.analysis.output_csv = str(system_dir / f"{name}_opa_motion_force.csv")
    c.analysis.displacement_png = str(system_dir / f"{name}_opa_displacement.png")
    c.analysis.force_png = str(system_dir / f"{name}_opa_force.png")
    return c


def run_all_systems_md(
    config: WorkflowInput,
    calc: Any,
    *,
    system_names: list[str] | None = None,
    results_dir: str | Path = "results",
    analyze: bool = True,
    skip_existing: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run MLIP-MD sequentially for selected packed systems."""
    import pandas as pd

    prepare_structure_input(config)
    selected = {
        _safe_name(name).lower()
        for name in system_names
    } if system_names is not None else None

    results_root = _path(config.workdir, results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, Any]] = {}
    summary_rows = []
    for solvent in config.structure.solvents:
        safe_name = _safe_name(solvent.name)
        if selected is not None and safe_name.lower() not in selected:
            continue

        system_config = configure_system_outputs(config, solvent, results_dir)
        input_xyz = _path(system_config.workdir, system_config.md.input_xyz)
        if not input_xyz.exists():
            raise FileNotFoundError(
                f"Missing packed structure for {solvent.name}: {input_xyz}. "
                "Run Packmol first with packed_xyz_files = run_packmol(config)."
            )

        print(f"\n=== Running MLIP-MD: {solvent.name} ===")
        print(f"Input: {system_config.md.input_xyz}")
        print(f"Trajectory: {system_config.md.output_traj}")
        traj_path = _path(system_config.workdir, system_config.md.output_traj)
        final_xyz_path = _path(system_config.workdir, system_config.md.output_xyz)

        if skip_existing and traj_path.exists() and final_xyz_path.exists():
            print(f"Skipping existing result: {solvent.name}")
            atoms = None
        else:
            atoms = run_mlip_md(system_config, calc)

        df = None
        if analyze:
            df = analyze_opa_motion(system_config)
            plot_opa_motion(system_config, df)
            summary_rows.append(
                {
                    "system": solvent.name,
                    "input_xyz": system_config.md.input_xyz,
                    "trajectory": system_config.md.output_traj,
                    "final_xyz": system_config.md.output_xyz,
                    "n_solvent_molecules": solvent.n_molecules,
                    "density_g_cm3": solvent.density_g_cm3,
                    "final_opa_disp_A": float(df["OPA_disp_norm_A"].iloc[-1]),
                    "mean_F_OPA_norm_eV_A": float(df["F_OPA_norm_eV_A"].mean()),
                    "max_F_OPA_norm_eV_A": float(df["F_OPA_norm_eV_A"].max()),
                }
            )

        outputs[solvent.name] = {
            "config": system_config,
            "atoms": atoms,
            "analysis": df,
            "trajectory": system_config.md.output_traj,
            "final_xyz": system_config.md.output_xyz,
            "csv": system_config.analysis.output_csv,
            "displacement_png": system_config.analysis.displacement_png,
            "force_png": system_config.analysis.force_png,
        }

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = results_root / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved batch summary: {summary_path}")

    return outputs


# =========================
# Backend run settings
# =========================

WORKDIR = "."
BOX_LENGTH_A = 45.0
SYSTEM_BOX_LENGTH_A = {
    "prol": 40.0,
}
PACKMOL_EXECUTABLE = "/home/jovyan/miniconda3/bin/packmol"

RUN_PACKMOL = True
RUN_MD = True
RUN_ANALYSIS = True
SKIP_EXISTING_PACKMOL = False
SKIP_EXISTING_MD = True
MD_MAX_ATOMS = 30000
MD_MAX_ESTIMATED_NEIGHBORS = 1650000
QUIET_PACKMOL = True

# None means run all five systems. Example for testing: ["acetone", "thf"]
SYSTEM_NAMES = None

RESULTS_DIR = "results"

MD_N_STEPS = 100000
MD_LOG_INTERVAL = 1000
MD_TEMPERATURE_K = 300.0
MD_TIMESTEP_FS = 1.0
MD_FRICTION_PER_FS = 0.01

# Choose "matlantis" or "pfp".
CALCULATOR_KIND = "matlantis"
MATLANTIS_MODEL_VERSION = "v9.0.0"
MATLANTIS_CALC_MODE = "R2SCAN"
PFP_CALC_MODE = "PBE_U_PLUS_D3"


def build_config() -> WorkflowInput:
    config = WorkflowInput(
        workdir=WORKDIR,
        packmol=PackmolInput(
            box_length=BOX_LENGTH_A,
            margin=2.0,
            tolerance=2.2,
            fixed_opa=True,
            opa_center=None,
        ),
        md=MDInput(
            n_steps=MD_N_STEPS,
            log_interval=MD_LOG_INTERVAL,
            temperature=MD_TEMPERATURE_K,
            timestep_fs=MD_TIMESTEP_FS,
            friction_per_fs=MD_FRICTION_PER_FS,
        ),
    )
    overrides = {_safe_name(name).lower(): box for name, box in SYSTEM_BOX_LENGTH_A.items()}
    for solvent in config.structure.solvents:
        solvent.box_length_a = overrides.get(_safe_name(solvent.name).lower())
    return config


def build_calculator():
    if CALCULATOR_KIND == "matlantis":
        return create_matlantis_calculator(
            model_version=MATLANTIS_MODEL_VERSION,
            calc_mode=MATLANTIS_CALC_MODE,
        )
    if CALCULATOR_KIND == "pfp":
        return create_pfp_calculator(calc_mode=PFP_CALC_MODE)
    raise ValueError(f"Unknown CALCULATOR_KIND: {CALCULATOR_KIND}")


def main() -> None:
    config = build_config()

    print("Preparing Packmol inputs...")
    prepare_structure_input(config)
    if RUN_MD:
        validate_md_system_sizes(config, MD_MAX_ATOMS, SYSTEM_NAMES)
        validate_mlip_neighbor_limit(config, MD_MAX_ESTIMATED_NEIGHBORS, SYSTEM_NAMES)
    save_config(config)
    packmol_inputs = write_packmol_inputs(config)

    print("\nGenerated Packmol input files:")
    for path in packmol_inputs:
        print("  ", path)

    print("\nMolecule counts by system:")
    for solvent in config.structure.solvents:
        box_length = get_solvent_box_length(config, solvent)
        n_atoms_total = estimate_system_atom_count(config, solvent)
        print(
            f"{solvent.name:10s}  box={box_length:g} A  "
            f"density={solvent.density_g_cm3:g} g/cm3  "
            f"M={solvent.molar_mass_g_mol:.3f} g/mol  N={solvent.n_molecules}  "
            f"atoms={n_atoms_total}  "
            f"estimated_neighbors={estimate_pfp_neighbor_count(config, solvent):,}"
        )

    if RUN_PACKMOL:
        print("\nRunning Packmol...")
        if shutil.which(PACKMOL_EXECUTABLE) is None and not Path(PACKMOL_EXECUTABLE).exists():
            print(
                "\nERROR: Packmol was not found.\n"
                f"Current PACKMOL_EXECUTABLE = {PACKMOL_EXECUTABLE!r}\n"
                "Set PACKMOL_EXECUTABLE to the full binary path, or install/load Packmol.\n"
                "The Packmol input files have already been generated in packmol_systems/."
            )
            return
        packed_xyz_files = run_packmol(
            config,
            executable=PACKMOL_EXECUTABLE,
            skip_existing=SKIP_EXISTING_PACKMOL,
            quiet=QUIET_PACKMOL,
        )
        print("\nPacked XYZ files:")
        for path in packed_xyz_files:
            print("  ", path)

    if RUN_MD:
        print("\nCreating MLIP calculator...")
        calc = build_calculator()
        print("Running MLIP-MD batch...")
        run_all_systems_md(
            config,
            calc,
            system_names=SYSTEM_NAMES,
            results_dir=RESULTS_DIR,
            analyze=RUN_ANALYSIS,
            skip_existing=SKIP_EXISTING_MD,
        )

    print("\nWorkflow finished.")


# Copying this whole file into a Jupyter cell should start the backend workflow.
main()
