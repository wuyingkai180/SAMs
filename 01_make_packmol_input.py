import math
from pathlib import Path

# =========================
# User settings
# =========================
box_length = 100.0  # Angstrom

density_g_cm3 = 0.82  # Ketone solvent density, g/cm^3

# Example: nonanone, C9H18O
# C: 12.011, H: 1.008, O: 15.999
solvent_molar_mass = 142.24  # g/mol, change for your ketone solvent

solvent_xyz = "ketone.xyz"
opa_xyz = "OPA.xyz"

output_xyz = "opa_ketone_box.xyz"
packmol_input = "packmol.inp"

# Keep molecules away from box boundary by this margin
margin = 2.0  # Angstrom

# =========================
# Calculate solvent number
# =========================
NA = 6.02214076e23

# 1 A^3 = 1e-24 cm^3
volume_cm3 = box_length**3 * 1e-24

# mass = density * volume
mass_g = density_g_cm3 * volume_cm3

# mol = mass / molar mass
n_mol = mass_g / solvent_molar_mass

# molecule number
n_solvent = int(round(n_mol * NA))

print(f"Box volume: {box_length**3:.3e} A^3")
print(f"Volume: {volume_cm3:.3e} cm^3")
print(f"Target density: {density_g_cm3} g/cm^3")
print(f"Solvent molar mass: {solvent_molar_mass} g/mol")
print(f"Estimated solvent molecules: {n_solvent}")

# =========================
# Write Packmol input
# =========================
inp = f"""
tolerance 2.2
filetype xyz
output {output_xyz}

structure {opa_xyz}
  number 1
  inside box {margin} {margin} {margin} {box_length - margin} {box_length - margin} {box_length - margin}
end structure

structure {solvent_xyz}
  number {n_solvent}
  inside box {margin} {margin} {margin} {box_length - margin} {box_length - margin} {box_length - margin}
end structure
"""

Path(packmol_input).write_text(inp.strip() + "\n")
print(f"Wrote {packmol_input}")
print(f"Packmol output will be {output_xyz}")
