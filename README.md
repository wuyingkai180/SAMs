# OPA in ketone solvent MLIP-MD scripts

This project contains only scripts for building and analyzing an OPA + ketone solvent box.

## Files

- `01_make_packmol_input.py`: generate `packmol.inp` from box size, solvent density, and molar mass.
- `02_run_packmol.sh`: run Packmol to generate `opa_ketone_box.xyz`.
- `03_run_mlip_md.py`: run 300 K MD using ASE + an MLIP calculator. Edit the calculator section before use.
- `04_analyze_opa_motion.py`: analyze OPA COM displacement and total net force on OPA.

## Required input files

Prepare these two files in the same directory:

- `ketone.xyz`
- `OPA.xyz`

Set `n_opa_atoms` in both `03_run_mlip_md.py` and `04_analyze_opa_motion.py` according to the first line of `OPA.xyz`.

## Basic workflow

```bash
python 01_make_packmol_input.py
bash 02_run_packmol.sh
python 03_run_mlip_md.py
python 04_analyze_opa_motion.py
```

Note: The force analyzed here is the total net force on OPA, not a strictly decomposed vdW-only force.
MD with a 100 x 100 x 100 A^3 box can be very large. Test with a smaller box first.
