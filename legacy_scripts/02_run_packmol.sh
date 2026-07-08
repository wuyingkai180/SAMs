#!/bin/bash
set -e

for input in packmol_systems/*_packmol.inp; do
  echo "Running Packmol: ${input}"
  packmol < "${input}"
done
