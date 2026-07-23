# OPA Multi-Solvent MLIP-MD Workflow

To build a fixed surface with solution added only in the upper vacuum layer,
use `packmol_solute_solvent_examples/packmol_surface_build.py`; its dedicated
instructions are in `packmol_solute_solvent_examples/README_surface.md`.
The insertion height is controlled explicitly by `surface_top_z_a`.

This project builds and simulates five independent OPA + molecule systems:

- `OPA + acetone`
- `OPA + n-heptane`
- `OPA + prol`
- `OPA + thf`
- `OPA + toluene`

`structures/opa.vasp` is inserted once per system. During Packmol modeling, OPA
is moved to the center of a 100 x 100 x 100 Angstrom box and fixed. Each other
molecule type is then packed around the fixed OPA according to its configured
density.




## Project Layout

```text
jupyter_copy_cell.py          # main self-contained Jupyter/backend workflow
structures/                   # original input structures
  opa.vasp
  Actone.vasp
  n-heptane.vasp
  prol.vasp
  thf.vasp
  Toluene.vasp
workflow_input.example.json   # editable example config
requirements.txt              # local Python dependencies
requirements-matlantis.txt    # Matlantis/PFP dependencies
legacy_scripts/               # archived split-script version
```

Generated files are written to `packmol_structures/`, `packmol_systems/`, and
`results/`.

## Files To Copy For Calculation

For a new Jupyter/Matlantis calculation directory, copy only these source files:

```text
jupyter_copy_cell.py
structures/
```

`jupyter_copy_cell.py` is the self-contained notebook script. It does not need
the archived split scripts in `legacy_scripts/`.

## Environment

Required for modeling and analysis:

```text
numpy
pandas
matplotlib
ase
packmol
```

Required for Matlantis/PFP MLIP-MD:

```text
pfp_api_client
matlantis_features
```

On Matlantis, the bundled Jupyter script is configured to call:

```python
PACKMOL_EXECUTABLE = "/home/jovyan/miniconda3/bin/packmol"
```

Change this path only if Packmol is installed somewhere else.

Check Packmol availability before a backend run:

```bash
which packmol
packmol < packmol_systems/acetone_packmol.inp
```

If `which packmol` returns nothing but Packmol exists in another environment,
set the absolute path in `jupyter_copy_cell.py`:

```python
PACKMOL_EXECUTABLE = "/path/to/packmol"
```

## Backend Script Workflow

If Jupyter cannot call Packmol reliably, submit the whole script as a backend
Python job instead:

```bash
python jupyter_copy_cell.py
```

For long jobs, run with unbuffered output and redirect the log:

```bash
python -u jupyter_copy_cell.py > backend_run.log 2>&1
```

The backend script uses the settings near the bottom of `jupyter_copy_cell.py`:

```python
RUN_PACKMOL = True
RUN_MD = True
RUN_ANALYSIS = True
SKIP_EXISTING_PACKMOL = True
SKIP_EXISTING_MD = True
SYSTEM_NAMES = None
CALCULATOR_KIND = "matlantis"
MATLANTIS_MODEL_VERSION = "v9.0.0"
MATLANTIS_CALC_MODE = "R2SCAN"
```

`SYSTEM_NAMES = None` runs all five systems. For a shorter test:

```python
SYSTEM_NAMES = ["acetone", "thf"]
```

## Jupyter Workflow

Open Jupyter in the directory containing `jupyter_copy_cell.py` and the
`structures/` folder, copy all of `jupyter_copy_cell.py` into one cell, then run
it. The final line calls `main()` automatically:

```python
main()
```

Step 1 generates Packmol inputs:

```python
prepare_structure_input(config)
save_config(config)
packmol_inputs = write_packmol_inputs(config)
```

Step 2 runs Packmol after you uncomment:

```python
packed_xyz_files = run_packmol(config)
```

Step 3 creates one calculator:

```python
calc = create_matlantis_calculator(
    model_version="v9.0.0",
    calc_mode="R2SCAN",
)
```

or:

```python
calc = create_pfp_calculator(calc_mode="PBE_U_PLUS_D3")
```

Step 4 runs all five systems sequentially:

```python
all_outputs = run_all_systems_md(
    config,
    calc,
    results_dir="results",
    analyze=True,
    skip_existing=True,
)
```

For a quick test on selected systems:

```python
all_outputs = run_all_systems_md(
    config,
    calc,
    system_names=["acetone", "thf"],
    results_dir="results",
    analyze=True,
    skip_existing=True,
)
```

## Generated Structure

After generating Packmol inputs:

```text
workflow_input.json
structures/
  opa.vasp
  Actone.vasp
  n-heptane.vasp
  prol.vasp
  thf.vasp
  Toluene.vasp
packmol_structures/
  opa_fixed.xyz
  Actone.xyz
  n-heptane.xyz
  prol.xyz
  thf.xyz
  Toluene.xyz
packmol_systems/
  acetone_packmol.inp
  n-heptane_packmol.inp
  prol_packmol.inp
  thf_packmol.inp
  toluene_packmol.inp
```

After running Packmol:

```text
packmol_systems/
  acetone_opa_box.xyz
  n-heptane_opa_box.xyz
  prol_opa_box.xyz
  thf_opa_box.xyz
  toluene_opa_box.xyz
```

After running all MD jobs:

```text
results/
  summary.csv
  acetone/
    acetone_md_300K.traj
    acetone_final_300K.xyz
    acetone_opa_motion_force.csv
    acetone_opa_displacement.png
    acetone_opa_force.png
  n-heptane/
    n-heptane_md_300K.traj
    n-heptane_final_300K.xyz
    n-heptane_opa_motion_force.csv
    n-heptane_opa_displacement.png
    n-heptane_opa_force.png
  prol/
  thf/
  toluene/
```

`results/summary.csv` contains per-system input/output paths, molecule counts,
final OPA displacement, and OPA force statistics.

## Notes

- The five systems are independent; the five molecule types are not mixed in
  one box.
- The density values are editable in `StructureInput.solvents`.
- `skip_existing=True` lets you resume a batch run without repeating systems
  that already have both trajectory and final XYZ files.
- The analyzed force is the total net force on OPA, not a decomposed vdW-only
  force.
