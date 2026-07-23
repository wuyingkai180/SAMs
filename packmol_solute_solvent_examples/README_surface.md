# Surface + Solution Packmol Builder

`packmol_surface_build.py` fixes a supplied surface structure in place and
adds one or more solvent components only in the vacuum region above it.

## Configure

Place the surface POSCAR/VASP file and solvent structure files in this folder,
then edit the `CONFIG` block in `packmol_surface_build.py`:

- `surface_file`: surface POSCAR/VASP filename;
- `surface_top_z_a`: Cartesian z coordinate (angstrom) of the highest surface
  layer, defined by the user;
- `surface_gap_a`: minimum vertical gap between that layer and the Packmol
  insertion region;
- `solution_top_z_a`: upper z boundary of the solution, or `None` to use
  `cell_z - top_margin_a`;
- `xy_margin_a`: margin from the x/y cell edges;
- `use_periodic_boundary_conditions`: emits Packmol's `pbc Lx Ly Lz`
  directive so molecules on opposite cell faces are checked using periodic
  distances; this requires Packmol 20.15.0 or newer;
- `solvents`: structures, densities, target volume fractions, and molecule
  counts, using the same style as `packmol_build.py`.
- `solutes`: optional solute structures and counts. By default a solute is
  packed anywhere in the common solution region. Set `z_min_a`/`z_max_a` to
  restrict it to a thinner layer, or set `fixed_center_a=[x, y, z]` to center
  and fix a single solute at an exact Cartesian position. Set
  `n_molecules=0` to keep the configuration entry while omitting that solute
  from the generated Packmol input, consistent with `packmol_build.py`.

The lower solution boundary is calculated as:

```text
surface_top_z_a + surface_gap_a
```

Thus, the common molecular placement box is
`[xy_margin, xy_margin, surface_top + gap]` to
`[Lx-xy_margin, Ly-xy_margin, solution_top]`. Solvents always use this box;
each solute may optionally use a narrower z interval inside it.

The script currently accepts an axis-aligned orthorhombic surface cell. This
restriction keeps Packmol's Cartesian `inside box` and `pbc` regions consistent
with the periodic VASP cell. For a periodic surface, leave `xy_margin_a=0.0`.

## Boundary conditions

The output VASP cell is periodic in x, y, and z. Choose the z arrangement that
matches the intended physical model:

1. **One-sided surface + solution + vacuum:** set `solution_top_z_a` below the
   cell top so that a vacuum gap remains above the liquid. The gap should be
   larger than the real-space interaction cutoff plus a safety margin. For
   electronic-structure calculations, also converge the vacuum thickness and
   use the code's slab/dipole correction when appropriate.
2. **Periodic solid/liquid stack:** allow the solution to reach the top only if
   it is intended to contact the bottom face of the next periodic slab.
3. **Symmetric slab:** put equivalent solution regions on both slab faces when
   a symmetric, zero-net-dipole setup is required. This script currently builds
   the one-sided region; a second region must be configured in an extended
   version.

Packmol PBC prevents initial solvent-solvent clashes across cell faces. It does
not remove the physical interaction between periodic images during the later
simulation; that interaction is controlled by the cell geometry and the chosen
MD/DFT boundary settings.

Surface x/y coordinates are wrapped automatically into the periodic cell.
Surface z coordinates must already lie between 0 and the cell z length; the
script stops with a clear error if the slab needs to be recentered in z.

## Run

```powershell
python .\packmol_surface_build.py
```

The script writes Packmol inputs, logs, and packed XYZ files under
`surface_systems/`. It also writes a `.vasp` version with the original surface
cell retained. Converted Packmol source structures are placed under
`surface_source_xyz/`, and the resolved configuration is saved as
`surface_solution_packmol_config.json`.

For input-generation only, set `runtime.run_packmol=False`.

The checked-in example is matched to `surface.vasp`: its highest atom is at
11.0115 A, the liquid occupies z=13.0115..35.0 A, and 15 A of vacuum remains
at the top of the 50 A cell. At the configured 10:90 volume composition this
region holds approximately 7 acetone and 33 n-heptane molecules at their input
bulk densities.
