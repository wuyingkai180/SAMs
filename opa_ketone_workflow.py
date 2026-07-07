from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


AVOGADRO = 6.02214076e23


@dataclass
class StructureInput:
    opa_xyz: str = "OPA.xyz"
    solvent_xyz: str = "ketone.xyz"
    n_opa_atoms: int | None = None
    n_solvent_atoms: int | None = None


@dataclass
class PackmolInput:
    box_length: float = 100.0
    density_g_cm3: float = 0.82
    solvent_molar_mass: float = 142.24
    margin: float = 2.0
    tolerance: float = 2.2
    packmol_input: str = "packmol.inp"
    output_xyz: str = "opa_ketone_box.xyz"


@dataclass
class MDInput:
    input_xyz: str = "opa_ketone_box.xyz"
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


def read_xyz_atom_count(xyz_file: str | Path) -> int:
    first_line = Path(xyz_file).read_text(encoding="utf-8").splitlines()[0].strip()
    try:
        return int(first_line)
    except ValueError as exc:
        raise ValueError(f"{xyz_file} does not look like a valid XYZ file.") from exc


def load_workflow_input(json_file: str | Path = "workflow_input.json") -> WorkflowInput:
    data = json.loads(Path(json_file).read_text(encoding="utf-8"))
    return workflow_input_from_dict(data)


def save_workflow_input(
    config: WorkflowInput,
    json_file: str | Path = "workflow_input.json",
) -> Path:
    path = Path(json_file)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return path


def workflow_input_from_dict(data: dict[str, Any]) -> WorkflowInput:
    return WorkflowInput(
        workdir=data.get("workdir", "."),
        structure=StructureInput(**data.get("structure", {})),
        packmol=PackmolInput(**data.get("packmol", {})),
        md=MDInput(**data.get("md", {})),
        analysis=AnalysisInput(**data.get("analysis", {})),
    )


def prepare_structure_input(config: WorkflowInput) -> WorkflowInput:
    """Fill atom counts from OPA and solvent XYZ files."""
    opa_path = _path(config.workdir, config.structure.opa_xyz)
    solvent_path = _path(config.workdir, config.structure.solvent_xyz)

    if not opa_path.exists():
        raise FileNotFoundError(f"Missing OPA structure: {opa_path}")
    if not solvent_path.exists():
        raise FileNotFoundError(f"Missing solvent structure: {solvent_path}")

    config.structure.n_opa_atoms = read_xyz_atom_count(opa_path)
    config.structure.n_solvent_atoms = read_xyz_atom_count(solvent_path)
    return config


def estimate_solvent_count(config: WorkflowInput) -> int:
    volume_cm3 = config.packmol.box_length**3 * 1e-24
    mass_g = config.packmol.density_g_cm3 * volume_cm3
    n_mol = mass_g / config.packmol.solvent_molar_mass
    return int(round(n_mol * AVOGADRO))


def write_packmol_input(config: WorkflowInput) -> Path:
    n_solvent = estimate_solvent_count(config)
    p = config.packmol
    s = config.structure

    if p.margin <= 0 or p.margin * 2 >= p.box_length:
        raise ValueError("Packmol margin must be positive and smaller than half the box length.")

    content = f"""
tolerance {p.tolerance}
filetype xyz
output {p.output_xyz}

structure {s.opa_xyz}
  number 1
  inside box {p.margin} {p.margin} {p.margin} {p.box_length - p.margin} {p.box_length - p.margin} {p.box_length - p.margin}
end structure

structure {s.solvent_xyz}
  number {n_solvent}
  inside box {p.margin} {p.margin} {p.margin} {p.box_length - p.margin} {p.box_length - p.margin} {p.box_length - p.margin}
end structure
""".strip()

    path = _path(config.workdir, p.packmol_input)
    path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    print(f"Estimated solvent molecules: {n_solvent}")
    print(f"Packmol output: {_path(config.workdir, p.output_xyz)}")
    return path


def run_packmol(config: WorkflowInput, executable: str = "packmol") -> Path:
    packmol_input = _path(config.workdir, config.packmol.packmol_input)
    output_xyz = _path(config.workdir, config.packmol.output_xyz)
    if not packmol_input.exists():
        write_packmol_input(config)

    with packmol_input.open("rb") as stdin:
        subprocess.run(
            [executable],
            cwd=config.workdir,
            stdin=stdin,
            check=True,
        )
    return output_xyz


def load_md_atoms(config: WorkflowInput):
    from ase.io import read

    input_xyz = _path(config.workdir, config.md.input_xyz)
    atoms = read(input_xyz)
    atoms.set_cell([config.packmol.box_length] * 3)
    atoms.set_pbc([True, True, True])

    n_opa_atoms = require_n_opa_atoms(config)
    atoms.info["opa_indices"] = list(range(n_opa_atoms))
    atoms.info["solvent_indices"] = list(range(n_opa_atoms, len(atoms)))
    return atoms


def require_n_opa_atoms(config: WorkflowInput) -> int:
    if config.structure.n_opa_atoms is None or config.structure.n_opa_atoms <= 0:
        prepare_structure_input(config)
    if config.structure.n_opa_atoms is None or config.structure.n_opa_atoms <= 0:
        raise ValueError("n_opa_atoms could not be determined from OPA.xyz.")
    return config.structure.n_opa_atoms


def run_mlip_md(config: WorkflowInput, calc: Any):
    from ase import units
    from ase.io import write
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
        ZeroRotation,
    )

    atoms = load_md_atoms(config)
    atoms.calc = calc
    n_opa_atoms = require_n_opa_atoms(config)
    opa_indices = list(range(n_opa_atoms))

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

    print("Starting MD...")
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

    traj_file = _path(config.workdir, config.analysis.traj_file)
    n_opa_atoms = require_n_opa_atoms(config)
    opa_indices = list(range(n_opa_atoms))
    frames = read(traj_file, index=":")

    times_ps = []
    opa_com_list = []
    opa_force_list = []
    opa_force_norm_list = []

    for i, atoms in enumerate(frames):
        time_ps = i * config.analysis.save_interval * config.analysis.dt_fs / 1000.0
        opa = atoms[opa_indices]
        com = opa.get_center_of_mass()
        forces = atoms.get_forces()
        f_opa = forces[opa_indices].sum(axis=0)

        times_ps.append(time_ps)
        opa_com_list.append(com)
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

    csv_path = _path(config.workdir, config.analysis.output_csv)
    df.to_csv(csv_path, index=False)
    print(f"Saved analysis to {csv_path}")
    print(f"Final OPA displacement: {disp_norm[-1]:.3f} A")
    print(f"Mean |F_OPA|: {np.mean(opa_force_norm_list):.6f} eV/A")
    print(f"Max  |F_OPA|: {np.max(opa_force_norm_list):.6f} eV/A")
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
    print(
        "Saved plots: "
        f"{config.analysis.displacement_png}, {config.analysis.force_png}"
    )


def write_example_input(json_file: str | Path = "workflow_input.json") -> Path:
    config = WorkflowInput()
    return save_workflow_input(config, json_file)


def notebook_quickstart() -> str:
    return """
from opa_ketone_workflow import *

config = WorkflowInput(
    packmol=PackmolInput(box_length=60.0, density_g_cm3=0.82, solvent_molar_mass=142.24),
    md=MDInput(n_steps=1000, log_interval=10),
)

prepare_structure_input(config)
write_packmol_input(config)
# run_packmol(config)

# Define your MLIP calculator in the notebook, then run:
# atoms = run_mlip_md(config, calc)
# df = analyze_opa_motion(config)
# plot_opa_motion(config, df)
""".strip()


if __name__ == "__main__":
    config = WorkflowInput()
    save_workflow_input(config)
    print("Wrote workflow_input.json")
    try:
        prepare_structure_input(config)
    except FileNotFoundError as exc:
        print(f"Skipped Packmol input generation: {exc}")
    else:
        save_workflow_input(config)
        write_packmol_input(config)
    print("Jupyter quickstart:")
    print(notebook_quickstart())
