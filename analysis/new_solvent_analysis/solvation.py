"""Total and species-resolved OPA solvation structure analysis."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from ase.io import read
from ase.io.trajectory import Trajectory

from .manifest import SystemRecord


R_MAX_A = 12.0
DR_A = 0.1
COORDINATION_CUTOFF_A = 4.0
N_OPA_ATOMS = 61
STRUCTURE_FILES = {
    "acetone": "Actone.vasp",
    "acetonitrile": "acetonitrile.vasp",
    "CPME": "CPME.vasp",
    "cyclohexane": "cyclohexane.vasp",
    "DMC": "DMC.vasp",
    "ethanol": "ethanol.vasp",
    "ethyl_acetate": "ethyl_acetate.vasp",
    "isopropanol": "isopropanol.vasp",
    "methanol": "methanol.vasp",
    "n-heptane": "n-heptane.vasp",
    "prol": "prol.vasp",
    "propylene_carbonate": "propylene_carbonate.vasp",
    "p-xylene": "p-xylene.vasp",
    "thf": "thf.vasp",
    "toluene": "Toluene.vasp",
}
BOND_CUTOFF_A = {
    frozenset(("P", "O")): 1.9,
    frozenset(("O", "H")): 1.3,
    frozenset(("C", "H")): 1.3,
    frozenset(("C", "C")): 1.75,
    frozenset(("P", "C")): 1.95,
}


@dataclass(frozen=True)
class OpaTopology:
    symbols: tuple[str, ...]
    head_indices: tuple[int, ...]
    terminal_carbon: int
    n_atoms: int


@dataclass(frozen=True)
class SpeciesAssignment:
    atom_indices: dict[str, np.ndarray]
    molecule_indices: dict[str, tuple[np.ndarray, ...]]
    molecule_counts: dict[str, int]


@dataclass(frozen=True)
class RadialProfile:
    r_A: np.ndarray
    g_head: np.ndarray
    g_tail: np.ndarray
    running_head: np.ndarray
    running_tail: np.ndarray


def enrichment_factor(local_count: float, local_total: float, bulk_fraction: float) -> float:
    if local_total <= 0 or bulk_fraction <= 0:
        return float("nan")
    return float((local_count / local_total) / bulk_fraction)


def minimum_image(displacement: np.ndarray, cell: np.ndarray) -> np.ndarray:
    displacement = np.asarray(displacement, dtype=float)
    cell = np.asarray(cell, dtype=float)
    inverse = np.linalg.inv(cell)
    fractional = displacement @ inverse
    fractional -= np.round(fractional)
    return fractional @ cell


def assign_species_blocks(
    symbols: Sequence[str],
    templates: dict[str, tuple[str, ...]],
    molecule_counts: dict[str, int],
) -> dict[str, list[int]]:
    """Assign a fixed-order atom stream to contiguous molecule blocks."""
    stream = tuple(symbols)
    cursor = 0
    indices: dict[str, list[int]] = {}
    for species, template in templates.items():
        if not template:
            raise ValueError(f"{species}: empty molecule template")
        count = molecule_counts.get(species, 0)
        if count < 1:
            raise ValueError(f"{species}: molecule count must be positive")
        expected = template * count
        actual = stream[cursor : cursor + len(expected)]
        if actual != expected:
            raise ValueError(
                f"{species}: trajectory symbol order does not match {count} template blocks"
            )
        indices[species] = list(range(cursor, cursor + len(expected)))
        cursor += len(expected)
    if cursor != len(stream):
        raise ValueError(f"species assignment consumed {cursor} of {len(stream)} atoms")
    return indices


def _infer_molecule_counts(
    symbols: tuple[str, ...], templates: dict[str, tuple[str, ...]]
) -> dict[str, int]:
    names = list(templates)
    if len(names) == 1:
        name = names[0]
        length = len(templates[name])
        if len(symbols) % length:
            raise ValueError(f"{name}: solvent atom count is not divisible by template length")
        counts = {name: len(symbols) // length}
        assign_species_blocks(symbols, templates, counts)
        return counts
    if len(names) != 2:
        raise ValueError("only one- and two-component solvent systems are supported")
    first, second = names
    first_template, second_template = templates[first], templates[second]
    candidates = []
    for first_count in range(1, len(symbols) // len(first_template) + 1):
        consumed = first_count * len(first_template)
        remaining = len(symbols) - consumed
        if remaining <= 0 or remaining % len(second_template):
            continue
        second_count = remaining // len(second_template)
        counts = {first: first_count, second: second_count}
        try:
            assign_species_blocks(symbols, templates, counts)
        except ValueError:
            continue
        candidates.append(counts)
    if len(candidates) != 1:
        raise ValueError(
            f"species molecule counts are not uniquely identifiable; candidates={candidates}"
        )
    return candidates[0]


def assign_species(
    record: SystemRecord, symbols: Sequence[str], structures_root: Path | None = None
) -> SpeciesAssignment:
    """Resolve solvent atoms and molecule blocks from fixed trajectory order."""
    structures_root = (
        Path(__file__).resolve().parents[2] / "structures"
        if structures_root is None
        else Path(structures_root)
    )
    solvent_symbols = tuple(symbols)
    templates: dict[str, tuple[str, ...]] = {}
    for component in record.components:
        try:
            structure_path = structures_root / STRUCTURE_FILES[component]
        except KeyError as exc:
            raise ValueError(f"No structure mapping for solvent {component!r}") from exc
        templates[component] = tuple(read(structure_path).get_chemical_symbols())
    counts = _infer_molecule_counts(solvent_symbols, templates)
    relative_atoms = assign_species_blocks(solvent_symbols, templates, counts)
    molecule_indices: dict[str, tuple[np.ndarray, ...]] = {}
    cursor = 0
    for species, template in templates.items():
        molecules = []
        for _ in range(counts[species]):
            molecules.append(np.arange(cursor, cursor + len(template), dtype=int))
            cursor += len(template)
        molecule_indices[species] = tuple(molecules)
    return SpeciesAssignment(
        atom_indices={key: np.asarray(value, dtype=int) for key, value in relative_atoms.items()},
        molecule_indices=molecule_indices,
        molecule_counts=counts,
    )


def classify_opa_head_tail(opa_path: Path) -> OpaTopology:
    atoms = read(opa_path)
    symbols = tuple(atoms.get_chemical_symbols())
    coordinates = atoms.get_positions()
    if len(symbols) != N_OPA_ATOMS:
        raise ValueError(f"{opa_path}: expected {N_OPA_ATOMS} OPA atoms; found {len(symbols)}")
    adjacency: list[list[int]] = [[] for _ in symbols]
    for left in range(len(symbols)):
        for right in range(left + 1, len(symbols)):
            cutoff = BOND_CUTOFF_A.get(frozenset((symbols[left], symbols[right])))
            if cutoff and np.linalg.norm(coordinates[left] - coordinates[right]) < cutoff:
                adjacency[left].append(right)
                adjacency[right].append(left)
    phosphorus = symbols.index("P")
    graph_distance = [-1] * len(symbols)
    graph_distance[phosphorus] = 0
    queue = deque([phosphorus])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if graph_distance[neighbor] == -1:
                graph_distance[neighbor] = graph_distance[node] + 1
                queue.append(neighbor)
    if any(distance < 0 for distance in graph_distance):
        raise ValueError(f"{opa_path}: disconnected OPA connectivity graph")
    oxygen = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    acidic_hydrogen = sorted(
        {
            neighbor
            for oxygen_index in oxygen
            for neighbor in adjacency[oxygen_index]
            if symbols[neighbor] == "H"
        }
    )
    head = tuple(sorted([phosphorus, *oxygen, *acidic_hydrogen]))
    carbon = [index for index, symbol in enumerate(symbols) if symbol == "C"]
    terminal = max(carbon, key=lambda index: graph_distance[index])
    return OpaTopology(symbols, head, terminal, len(symbols))


def _profile_from_histograms(
    head_hist: np.ndarray,
    tail_hist: np.ndarray,
    n_frames: int,
    n_head: int,
    n_atoms: int,
    mean_volume: float,
) -> RadialProfile:
    edges = np.arange(0.0, R_MAX_A + DR_A, DR_A)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    shell_volume = 4.0 * math.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = n_atoms / mean_volume
    g_head = head_hist / (n_frames * n_head * shell_volume * density)
    g_tail = tail_hist / (n_frames * shell_volume * density)
    return RadialProfile(
        r_A=r_mid,
        g_head=g_head,
        g_tail=g_tail,
        running_head=np.cumsum(head_hist) / (n_frames * n_head),
        running_tail=np.cumsum(tail_hist) / n_frames,
    )


def _first_peak(profile: RadialProfile, values: np.ndarray) -> tuple[float, float]:
    start = int(1.5 / DR_A)
    index = start + int(np.argmax(values[start:]))
    return float(profile.r_A[index]), float(values[index])


def _coordination(profile: RadialProfile, running: np.ndarray) -> float:
    return float(np.interp(COORDINATION_CUTOFF_A, profile.r_A, running))


def compute_solvation_metrics(
    record: SystemRecord, opa_path: Path
) -> tuple[dict[str, float | str | int], pd.DataFrame]:
    """Compute total/species RDF, CN, and molecule-based enrichment for one run."""
    topology = classify_opa_head_tail(opa_path)
    with Trajectory(str(record.traj_path), mode="r") as trajectory:
        first = trajectory[0]
        symbols = first.get_chemical_symbols()
        if tuple(symbols[: topology.n_atoms]) != topology.symbols:
            raise ValueError(f"{record.traj_path}: OPA is not the first 61-atom block")
        assignment = assign_species(record, symbols[topology.n_atoms :])
        species_names = list(assignment.atom_indices)
        profile_names = ["all", *species_names]
        n_bins = int(R_MAX_A / DR_A)
        head_hist = {name: np.zeros(n_bins, dtype=float) for name in profile_names}
        tail_hist = {name: np.zeros(n_bins, dtype=float) for name in profile_names}
        local_head = {name: 0.0 for name in species_names}
        local_tail = {name: 0.0 for name in species_names}
        volumes = []
        for atoms in trajectory:
            positions = atoms.get_positions()
            cell = np.asarray(atoms.cell.array, dtype=float)
            volumes.append(float(abs(np.linalg.det(cell))))
            solvent_positions = positions[topology.n_atoms :]
            selections = {"all": np.arange(len(solvent_positions), dtype=int)}
            selections.update(assignment.atom_indices)
            for species, indices in selections.items():
                selected = solvent_positions[indices]
                head_diff = selected[None, :, :] - positions[list(topology.head_indices)][:, None, :]
                head_dist = np.linalg.norm(minimum_image(head_diff, cell), axis=-1)
                tail_diff = selected - positions[topology.terminal_carbon]
                tail_dist = np.linalg.norm(minimum_image(tail_diff, cell), axis=-1)
                head_hist[species] += np.histogram(
                    head_dist.ravel(), bins=n_bins, range=(0.0, R_MAX_A)
                )[0]
                tail_hist[species] += np.histogram(
                    tail_dist, bins=n_bins, range=(0.0, R_MAX_A)
                )[0]
            for species, molecules in assignment.molecule_indices.items():
                for molecule in molecules:
                    molecule_positions = solvent_positions[molecule]
                    head_diff = molecule_positions[None, :, :] - positions[
                        list(topology.head_indices)
                    ][:, None, :]
                    tail_diff = molecule_positions - positions[topology.terminal_carbon]
                    if np.any(
                        np.linalg.norm(minimum_image(head_diff, cell), axis=-1)
                        <= COORDINATION_CUTOFF_A
                    ):
                        local_head[species] += 1.0
                    if np.any(
                        np.linalg.norm(minimum_image(tail_diff, cell), axis=-1)
                        <= COORDINATION_CUTOFF_A
                    ):
                        local_tail[species] += 1.0
        n_frames = len(trajectory)
    total_solvent_atoms = sum(len(value) for value in assignment.atom_indices.values())
    profiles: dict[str, RadialProfile] = {}
    rows = []
    for species in profile_names:
        n_atoms = total_solvent_atoms if species == "all" else len(assignment.atom_indices[species])
        profile = _profile_from_histograms(
            head_hist[species], tail_hist[species], n_frames,
            len(topology.head_indices), n_atoms, float(np.mean(volumes))
        )
        profiles[species] = profile
        rows.extend(
            {
                "group": record.group,
                "system": record.system,
                "species": species,
                "r_A": float(radius),
                "g_head": float(g_head),
                "g_tail": float(g_tail),
                "running_head": float(running_head),
                "running_tail": float(running_tail),
            }
            for radius, g_head, g_tail, running_head, running_tail in zip(
                profile.r_A, profile.g_head, profile.g_tail,
                profile.running_head, profile.running_tail
            )
        )
    total_profile = profiles["all"]
    head_peak_r, head_peak_g = _first_peak(total_profile, total_profile.g_head)
    tail_peak_r, tail_peak_g = _first_peak(total_profile, total_profile.g_tail)
    head_coord = _coordination(total_profile, total_profile.running_head)
    tail_coord = _coordination(total_profile, total_profile.running_tail)
    metrics: dict[str, float | str | int] = {
        "group": record.group,
        "system": record.system,
        "n_trajectory_frames": n_frames,
        "n_solvent_atoms": total_solvent_atoms,
        "mean_volume_A3": float(np.mean(volumes)),
        "head_first_peak_r_A": head_peak_r,
        "head_first_peak_g": head_peak_g,
        "head_coord_number_4A": head_coord,
        "tail_first_peak_r_A": tail_peak_r,
        "tail_first_peak_g": tail_peak_g,
        "tail_coord_number_4A": tail_coord,
        "head_tail_coord_ratio": head_coord / tail_coord if tail_coord else float("nan"),
    }
    total_molecules = sum(assignment.molecule_counts.values())
    mean_local_head_total = sum(local_head.values()) / n_frames
    mean_local_tail_total = sum(local_tail.values()) / n_frames
    for species in species_names:
        profile = profiles[species]
        bulk_fraction = assignment.molecule_counts[species] / total_molecules
        mean_head = local_head[species] / n_frames
        mean_tail = local_tail[species] / n_frames
        suffix = species.replace("-", "_")
        metrics[f"molecule_count__{suffix}"] = assignment.molecule_counts[species]
        metrics[f"bulk_molecule_fraction__{suffix}"] = bulk_fraction
        metrics[f"head_coord_number_4A__{suffix}"] = _coordination(
            profile, profile.running_head
        )
        metrics[f"tail_coord_number_4A__{suffix}"] = _coordination(
            profile, profile.running_tail
        )
        metrics[f"head_local_molecules_4A__{suffix}"] = mean_head
        metrics[f"tail_local_molecules_4A__{suffix}"] = mean_tail
        metrics[f"head_enrichment_4A__{suffix}"] = enrichment_factor(
            mean_head, mean_local_head_total, bulk_fraction
        )
        metrics[f"tail_enrichment_4A__{suffix}"] = enrichment_factor(
            mean_tail, mean_local_tail_total, bulk_fraction
        )
    return metrics, pd.DataFrame(rows)
