# OPA in ketone solvent MLIP-MD workflow

This project builds and analyzes an OPA + ketone solvent box for ASE-based
MLIP molecular dynamics.

The main entry point is now:

- `opa_ketone_workflow.py`

It contains the structure input, Packmol input generation, MD runner, and OPA
motion/force analysis functions. The numbered scripts are thin wrappers kept
for command-line use.

## Required input files

Put these XYZ files in this directory:

- `OPA.xyz`
- `ketone.xyz`

The OPA atom count is read automatically from the first line of `OPA.xyz`.

## Jupyter usage

Use the workflow from one Python file:

```python
from opa_ketone_workflow import *

config = WorkflowInput(
    packmol=PackmolInput(
        box_length=60.0,
        density_g_cm3=0.82,
        solvent_molar_mass=142.24,
    ),
    md=MDInput(
        n_steps=1000,
        log_interval=10,
    ),
)

prepare_structure_input(config)
save_workflow_input(config)
write_packmol_input(config)
```

Run Packmol:

```python
run_packmol(config)
```

Then define the MLIP calculator in the notebook:

```python
# Example only. Replace with your actual calculator.
# from mace.calculators import MACECalculator
# calc = MACECalculator(model_paths="mace_model.model", device="cuda")

atoms = run_mlip_md(config, calc)
```

Analyze the trajectory:

```python
df = analyze_opa_motion(config)
plot_opa_motion(config, df)
df.head()
```

## Structured input file

Running:

```bash
python opa_ketone_workflow.py
```

creates:

- `workflow_input.json`
- `packmol.inp`

You can also start from `workflow_input.example.json`.

You can edit `workflow_input.json` and load it in Jupyter:

```python
from opa_ketone_workflow import *

config = load_workflow_input("workflow_input.json")
prepare_structure_input(config)
```

## Command-line workflow

The original numbered workflow still works:

```bash
python 01_make_packmol_input.py
bash 02_run_packmol.sh
python 03_run_mlip_md.py
python 04_analyze_opa_motion.py
```

Before running `03_run_mlip_md.py`, edit it and define `calc`.

## Outputs

- `packmol.inp`: Packmol input file.
- `opa_ketone_box.xyz`: packed OPA + solvent box.
- `md_300K.traj`: ASE trajectory.
- `final_300K.xyz`: final MD structure.
- `opa_motion_force.csv`: OPA COM displacement and total net force.
- `opa_displacement.png`: OPA COM displacement plot.
- `opa_force.png`: total OPA force norm plot.

The analyzed force is the total net force on OPA, not a decomposed vdW-only
force. A 100 x 100 x 100 Angstrom box can be very large; test with a smaller
box first.
