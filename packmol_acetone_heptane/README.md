# Acetone/n-Heptane Packmol Builder

This folder is self-contained for building three Packmol examples. It contains
no MD, energy export, trajectory analysis, or plotting code.

Files required in this directory:

```text
packmol_build.py
opa.vasp
Actone.vasp
n-heptane.vasp
thf.vasp
```

Running the script builds these three systems:

- one OPA in pure n-heptane;
- one OPA in 10:90 v/v acetone/n-heptane;
- two OPA molecules in 10:80:10 v/v acetone/n-heptane/THF.

When two OPA molecules are requested, Packmol places both inside the box before
adding the solvent components; they are not fixed at the same coordinates.

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
bottom of `packmol_build.py`. With an activated environment,
`packmol_executable="packmol"` uses that environment's executable. It can also
be replaced by an absolute path.

Generated inputs and structures are written under `packmol_systems/`;
converted molecular XYZ files are written under `packmol_structures/`.
