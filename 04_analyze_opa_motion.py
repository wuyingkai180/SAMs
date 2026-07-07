from ase.io import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# User settings
# =========================
traj_file = "md_300K.traj"
n_opa_atoms = 0  # Same as in 03_run_mlip_md.py
dt_fs = 1.0
save_interval = 100  # Same as log_interval in 03_run_mlip_md.py

output_csv = "opa_motion_force.csv"

if n_opa_atoms <= 0:
    raise ValueError("Please set n_opa_atoms manually.")

opa_indices = list(range(n_opa_atoms))

# =========================
# Read trajectory
# =========================
frames = read(traj_file, index=":")

times_ps = []
opa_com_list = []
opa_force_list = []
opa_force_norm_list = []

for i, atoms in enumerate(frames):
    time_ps = i * save_interval * dt_fs / 1000.0

    opa = atoms[opa_indices]
    com = opa.get_center_of_mass()

    # Total force on OPA.
    # This is the total net force on OPA, not a strictly decomposed vdW-only force.
    forces = atoms.get_forces()
    f_opa = forces[opa_indices].sum(axis=0)
    f_opa_norm = np.linalg.norm(f_opa)

    times_ps.append(time_ps)
    opa_com_list.append(com)
    opa_force_list.append(f_opa)
    opa_force_norm_list.append(f_opa_norm)

opa_com_arr = np.array(opa_com_list)
opa_force_arr = np.array(opa_force_list)

disp = opa_com_arr - opa_com_arr[0]
disp_norm = np.linalg.norm(disp, axis=1)

df = pd.DataFrame({
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
})

df.to_csv(output_csv, index=False)
print(f"Saved analysis to {output_csv}")

print()
print("Summary:")
print(f"Final OPA displacement: {disp_norm[-1]:.3f} A")
print(f"Mean |F_OPA|: {np.mean(opa_force_norm_list):.6f} eV/A")
print(f"Max  |F_OPA|: {np.max(opa_force_norm_list):.6f} eV/A")

plt.figure()
plt.plot(df["time_ps"], df["OPA_disp_norm_A"])
plt.xlabel("Time / ps")
plt.ylabel("OPA COM displacement / A")
plt.tight_layout()
plt.savefig("opa_displacement.png", dpi=300)

plt.figure()
plt.plot(df["time_ps"], df["F_OPA_norm_eV_A"])
plt.xlabel("Time / ps")
plt.ylabel("|F_OPA| / eV A$^{-1}$")
plt.tight_layout()
plt.savefig("opa_force.png", dpi=300)

print("Saved plots: opa_displacement.png, opa_force.png")
