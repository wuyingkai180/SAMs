# Generic Solute/Solvent Packmol Builder

This folder is self-contained for building three Packmol examples. It contains
no MD, energy export, trajectory analysis, or plotting code.

Files required in this directory:

```text
packmol_build.py
solute_1.vasp
Actone.vasp
n-heptane.vasp
thf.vasp
```

Running the script builds these three systems:

- one `solute_1` molecule in pure `solvent_1`;
- one `solute_1` molecule in 10:90 v/v `solvent_1`/`solvent_2`;
- two `solute_1` molecules in 10:80:10 v/v
  `solvent_1`/`solvent_2`/`solvent_3`.

Short output identifiers are used to keep filenames manageable:

```text
ex1
ex2
ex3
```

The final packed structures are therefore named `ex1.xyz`, `ex2.xyz`, and
`ex3.xyz`. Their Packmol inputs and logs use the same stems with `.inp` and
`.log` extensions.

The current chemical mapping is retained in `CONFIG` as metadata:

- `solute_1`: OPA (`solute_1.vasp`);
- pure-system `solvent_1`: n-heptane;
- binary/ternary `solvent_1`: acetone;
- binary/ternary `solvent_2`: n-heptane;
- ternary `solvent_3`: THF.

When two `solute_1` molecules are requested, Packmol places both inside the box
before adding the solvents; they are not fixed at the same coordinates.

Run:

```bash
python packmol_build.py
```

## Local Python virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy packmol
Get-Command packmol
python .\packmol_build.py
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy packmol
which packmol
python packmol_build.py
```

All editable settings are in the `CONFIG = WorkflowInput(...)` block near the
top of `packmol_build.py`. With an activated environment,
`packmol_executable="packmol"` uses that environment's executable. It can also
be replaced by an absolute path.

The script automatically uses its own directory as `workdir`, so it may be
started from any current directory. Packmol lookup checks the configured value,
the `PACKMOL_EXECUTABLE` environment variable, `PATH`, and the active Python
environment's `Scripts`, `Library/bin`, and `bin` directories. To confirm that
the correct Python environment can see Packmol, run:

```powershell
python -c "import shutil,sys; print(sys.executable); print(shutil.which('packmol'))"
```

After every Packmol run, the script verifies that the packed XYZ exists and
that its declared atom count matches the configured system.

Generated inputs and packed structures are written under `systems/`; converted
source XYZ files are written under `source_xyz/`. The saved
configuration is `solute_solvent_packmol_config.json`.
