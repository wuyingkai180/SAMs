# Safe Eight-System Resume Design

## Goal

Resume the solvent batch after completed ethanol and methanol calculations without
re-running their MD jobs, prevent the known acetonitrile neighbor-limit failure,
and report comparable expected neighbor counts for the remaining eight systems.

## Selected systems

The backend selection is explicitly limited to:

1. acetonitrile
2. propylene_carbonate
3. isopropanol
4. CPME
5. ethyl_acetate
6. DMC
7. p-xylene
8. cyclohexane

The selection applies consistently to system-size validation, Packmol generation,
Packmol execution, and MLIP-MD execution. Existing ethanol and methanol Packmol
structures and MD results are left untouched.

## Neighbor-limit correction

The empirical neighbor-density factor is recalibrated from the observed
acetonitrile request:

- box length: 59 Å
- atoms: 14,269
- observed neighbors: 1,709,819

This replaces the earlier factor, which predicted 1,649,470 neighbors and
underestimated this system by about 3.66%. The preflight limit remains 1,650,000,
providing margin below the server maximum of 1,700,001.

The acetonitrile box length is reduced from 59 Å to 58 Å. Its Packmol structure
must therefore be regenerated before MD.

## Packmol resume behavior

`write_packmol_inputs` and `run_packmol` accept the same optional system-name
selection used by the MD batch. When a selection is supplied, only selected
systems have inputs generated or Packmol executed.

For this resume run, Packmol overwrites the selected eight packed structures so
the 58 Å acetonitrile configuration cannot accidentally reuse its old 59 Å XYZ.
Ethanol and methanol are outside the selection and remain unchanged.

## Reporting

The existing molecule-count report is filtered to selected systems and prints
the recalibrated estimated neighbor count. A comparison table is also calculated
outside the remote API using the same configuration logic.

## Tests and verification

Automated tests load the script without executing `main()` and verify:

- the selected system list contains exactly the final eight systems;
- ethanol and methanol are excluded from selected Packmol inputs and runs;
- acetonitrile uses a 58 Å box;
- the calibrated estimate reproduces the observed 59 Å acetonitrile neighbor
  count used for calibration;
- all selected systems are checked against the 1,650,000 preflight limit.

Verification also includes Python compilation and a local dry calculation of the
eight expected neighbor counts. No Matlantis/PFP API request is made.
