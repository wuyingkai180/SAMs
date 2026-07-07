from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.langevin import Langevin
from ase import units
from ase.io.trajectory import Trajectory
import numpy as np

# =========================
# User settings
# =========================
input_xyz = "opa_ketone_box.xyz"
output_traj = "md_300K.traj"
output_xyz = "final_300K.xyz"

box_length = 100.0  # Angstrom

temperature = 300.0  # K
timestep_fs = 1.0
friction = 0.01  # Langevin friction parameter

n_steps = 100000  # 100 ps if dt = 1 fs
log_interval = 100

# Number of atoms in OPA molecule.
# Important: OPA was placed as the first molecule in Packmol input.
n_opa_atoms = 0  # Set this manually after checking OPA.xyz first line.

# =========================
# Read system
# =========================
atoms = read(input_xyz)

atoms.set_cell([box_length, box_length, box_length])
atoms.set_pbc([True, True, True])

if n_opa_atoms <= 0:
    raise ValueError(
        "Please set n_opa_atoms manually. "
        "For example, if OPA.xyz has 58 atoms, set n_opa_atoms = 58."
    )

opa_indices = list(range(n_opa_atoms))
solvent_indices = list(range(n_opa_atoms, len(atoms)))

atoms.info["opa_indices"] = opa_indices
atoms.info["solvent_indices"] = solvent_indices

# =========================
# MLIP calculator
# =========================
# Option 1: MACE
# from mace.calculators import MACECalculator
# calc = MACECalculator(
#     model_paths="mace_model.model",
#     device="cuda",
#     default_dtype="float64"
# )

# Option 2: Matlantis
# from pfp_api_client.pfp.estimator import Estimator
# from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
# estimator = Estimator(calc_mode="CRYSTAL_U0")
# calc = ASECalculator(estimator)

# Option 3: DeepMD
# from deepmd.calculator import DP
# calc = DP(model="frozen_model.pb")

raise RuntimeError("Please set your MLIP calculator in the calculator section.")

atoms.calc = calc

# =========================
# Initial velocities
# =========================
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
Stationary(atoms)
ZeroRotation(atoms)

# =========================
# MD
# =========================
dyn = Langevin(
    atoms,
    timestep_fs * units.fs,
    temperature_K=temperature,
    friction=friction / units.fs,
)

traj = Trajectory(output_traj, "w", atoms)


def print_status():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = atoms.get_temperature()
    etot = epot + ekin

    forces = atoms.get_forces()
    f_opa = forces[opa_indices].sum(axis=0)
    f_opa_norm = np.linalg.norm(f_opa)

    step = dyn.get_number_of_steps()

    print(
        f"Step {step:8d} | "
        f"T = {temp:8.2f} K | "
        f"Epot = {epot:14.6f} eV | "
        f"Etot = {etot:14.6f} eV | "
        f"|F_OPA_total| = {f_opa_norm:12.6f} eV/A"
    )


dyn.attach(traj.write, interval=log_interval)
dyn.attach(print_status, interval=log_interval)

print("Starting MD...")
dyn.run(n_steps)

traj.close()
write(output_xyz, atoms)

print("MD finished.")
print(f"Trajectory saved to {output_traj}")
print(f"Final structure saved to {output_xyz}")
