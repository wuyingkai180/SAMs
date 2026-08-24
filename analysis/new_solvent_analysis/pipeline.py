"""Deterministic, resumable orchestration for the corrected three-group dataset."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ase.io.trajectory import Trajectory

from .comparisons import (
    SOLVENT_CLASS,
    build_group1_comparison,
    build_group2_comparison,
    build_group3_comparison,
    collapse_duplicate_trajectories,
)
from .core_metrics import block_sensitivity, compute_core_metrics
from .manifest import SystemRecord, build_manifest, write_manifest
from .reporting import (
    render_group_reports,
    render_groupwise_index,
)
from .solvation import compute_solvation_metrics


SOLVATION_BASE_COLUMNS = (
    "head_first_peak_r_A", "head_first_peak_g", "head_coord_number_4A",
    "tail_first_peak_r_A", "tail_first_peak_g", "tail_coord_number_4A",
    "head_tail_coord_ratio", "mean_volume_A3", "n_solvent_atoms",
)
PROFILE_COLUMNS = (
    "group", "system", "species", "r_A", "g_head", "g_tail",
    "running_head", "running_tail",
)


@dataclass(frozen=True)
class RunSummary:
    system_count: int
    excluded_trash_systems: int
    output_root: Path | None
    report_paths: tuple[Path, ...]
    status: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(records: list[SystemRecord], literature_path: Path) -> str:
    payload = {
        "inputs": [(record.group, record.system, record.sha256) for record in records],
        "literature_sha256": _sha256(literature_path),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "pandas", "matplotlib", "ase", "scikit-learn", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _load_literature(path: Path) -> list[dict]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < 6:
        raise ValueError("literature evidence bank must contain at least six records")
    required = {"title", "doi", "url", "method_used", "limitations"}
    for index, record in enumerate(records):
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"literature record {index} missing {missing}")
    return records


def _existing_summary(output_root: Path, fingerprint: str) -> RunSummary:
    metadata_path = output_root / "analysis_run.json"
    if not metadata_path.exists():
        raise FileExistsError(f"existing output is incomplete: {output_root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("input_fingerprint") != fingerprint:
        raise FileExistsError("--resume refused: completed output does not match current inputs")
    reports = tuple(output_root / path for path in metadata.get("report_paths", []))
    return RunSummary(metadata["system_count"], 0, output_root, reports, "complete")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _collect_checkpoints(checkpoint_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_files = sorted(checkpoint_root.glob("*/metrics.json"))
    metrics = pd.DataFrame(
        [json.loads(path.read_text(encoding="utf-8")) for path in metric_files]
    )
    blocks = pd.concat(
        [pd.read_csv(path.parent / "blocks.csv") for path in metric_files], ignore_index=True
    )
    profile_frames = [pd.read_csv(path.parent / "profiles.csv") for path in metric_files]
    profiles = pd.concat(
        [frame for frame in profile_frames if not frame.empty], ignore_index=True
    )
    return metrics, blocks, profiles


def _headline_findings(metrics: pd.DataFrame) -> list[str]:
    group1 = metrics.loc[metrics["group"] == "group1"].sort_values("cosolvent_fraction")
    g1_best = group1.loc[group1["alpha"].idxmax()]
    g1_corr = group1[["cosolvent_fraction", "alpha"]].corr().iloc[0, 1]
    g1_force_corr = group1[["cosolvent_fraction", "mean_force_eV_A"]].corr().iloc[0, 1]
    group2 = metrics.loc[metrics["group"] == "group2"]
    g2_mobile = group2.loc[group2["alpha"].idxmax()]
    g2_low_force = group2.loc[group2["mean_force_eV_A"].idxmin()]
    class_means = group2.groupby("solvent_class")[["alpha", "mean_force_eV_A"]].mean()
    group3 = metrics.loc[
        (metrics["group"] == "group3")
        & (metrics["role"] == "n-heptane-base-composition")
    ]
    g3_head = group3.loc[group3["head_coord_number_4A"].idxmax()]
    enrichments = []
    for row in group3.itertuples():
        suffix = str(row.cosolvent).replace("-", "_")
        value = getattr(row, f"head_enrichment_4A__{suffix}", float("nan"))
        if pd.notna(value):
            enrichments.append((str(row.cosolvent), float(value)))
    enrichment_text = "、".join(f"{name}={value:.2f}" for name, value in enrichments)
    return [
        (
            f"【直接数据】Group 1 中 α 最高的是 {g1_best['system']} "
            f"(α={g1_best['alpha']:.3f})，30% 时出现明显下降，整体并非单调；"
            f"浓度与 α 的 Pearson r={g1_corr:.3f}，而与平均受力的 r={g1_force_corr:.3f}。"
        ),
        (
            f"【直接数据】Group 2 中 α 最高的是 {g2_mobile['system']} "
            f"(α={g2_mobile['alpha']:.3f})；平均受力最低的是 {g2_low_force['system']} "
            f"({g2_low_force['mean_force_eV_A']:.3f} eV Å⁻¹)。"
        ),
        (
            "【描述性类别均值】纯溶剂的 α（非质子极性/非极性/质子型）为 "
            f"{class_means.loc['aprotic-polar', 'alpha']:.3f}/"
            f"{class_means.loc['nonpolar', 'alpha']:.3f}/"
            f"{class_means.loc['protic', 'alpha']:.3f}；对应平均受力为 "
            f"{class_means.loc['aprotic-polar', 'mean_force_eV_A']:.3f}/"
            f"{class_means.loc['nonpolar', 'mean_force_eV_A']:.3f}/"
            f"{class_means.loc['protic', 'mean_force_eV_A']:.3f} eV Å⁻¹。"
        ),
        (
            f"【直接数据】Group 3 的四个 n-heptane 基混合物中，4 Å 头部配位数最高的是 "
            f"{g3_head['system']} ({g3_head['head_coord_number_4A']:.3f})；按实际分子数归一的"
            f"头部富集因子为 {enrichment_text}（1 表示与体相比例相同）。"
        ),
        "【推断】THF 和甲苯在本轨迹中优先出现在 OPA 头部附近，而丙酮和异丙醇相对贫化；动力学、受力和局部配位仍应联合判断，任何单一排名都不等同于成膜优劣。",
    ]


def _write_comparisons(metrics: pd.DataFrame, root: Path) -> list[Path]:
    outputs = []
    comparisons = {
        "group1": build_group1_comparison(metrics),
        "group2": build_group2_comparison(metrics),
        "group3": build_group3_comparison(metrics),
    }
    for group, result in comparisons.items():
        group_root = root / group
        tables = {
            "core_rows": result.core_rows,
            "correlations": result.correlations,
            "alternate_base_rows": result.alternate_base_rows,
            "pure_reference_rows": result.pure_reference_rows,
            **result.summary_tables,
        }
        for name, table in tables.items():
            if table is not None and not table.empty:
                path = group_root / f"{name}.csv"
                _write_frame(table, path)
                outputs.append(path)
        notes_path = group_root / "comparison_notes.json"
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(json.dumps(result.notes, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(notes_path)
    return outputs


def build_report_context(metrics: pd.DataFrame) -> dict:
    return {
        "system_count": len(metrics),
        "headline_findings": _headline_findings(metrics),
        "literature_findings": [
            "[Dietrich 等的磷酸分子/氧化铝模拟](https://doi.org/10.1039/C6CP08681K)提示极性与非极性溶剂会改变自组织路径，本项目据此分开考察头部与尾部溶剂化。",
            "[Kepten 等的短轨迹 TAMSD 指南](https://doi.org/10.1371/journal.pone.0117722)要求限制拟合区间并报告拟合质量；本项目保留 α、R² 和分块敏感性。",
            "[Hu 等的溶剂 RDF 方法](https://doi.org/10.3390/molecules23040733)支持用 RDF 与局部配位描述端基溶剂化，但这些量不替代表面覆盖率、倾角或缺陷表征。",
        ],
        "limitations": [
            "每个条件只有单条轨迹，分块值也不是独立重复。",
            "轨迹仅 10 ps，扩散指数和排名属于短时描述。",
            "模拟描述预吸附溶剂化，体系没有显式基底，不能直接判断 SAM 生长质量。",
            "Group 3 仅四个 n-heptane 基混合物进入核心横向比较；其余为分层参考。",
            "thf_toluene 的 CSV 为 101 帧而结构轨迹为 60 帧且帧 1 起坐标不一致；保留 CSV 动力学，排除该体系的轨迹 RDF/CN。",
        ],
        "links": {
            "三组系统指标": "../tables/system_metrics.csv",
            "RDF 数据": "../tables/rdf_profiles.csv",
            "文献方法矩阵": "../literature/literature_matrix.csv",
            "Group 1 报告": "group1.md",
            "Group 2 报告": "group2.md",
            "Group 3 报告": "group3.md",
        },
    }


def run_analysis(
    data_root: Path,
    output_root: Path,
    literature_path: Path,
    dry_run: bool = False,
    resume: bool = False,
) -> RunSummary:
    """Validate, compute, report, and atomically publish the analysis result tree."""
    data_root, output_root, literature_path = map(Path, (data_root, output_root, literature_path))
    records = build_manifest(data_root)
    literature = _load_literature(literature_path)
    fingerprint = _fingerprint(records, literature_path)
    if dry_run:
        return RunSummary(len(records), 0, None, (), "validated")
    if output_root.exists():
        if resume:
            return _existing_summary(output_root, fingerprint)
        raise FileExistsError(f"refusing to replace existing output: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp"
    metadata_path = temporary / "analysis_run.json"
    if temporary.exists():
        if not resume or not metadata_path.exists():
            raise FileExistsError(f"temporary run exists; use --resume after inspection: {temporary}")
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old_metadata.get("input_fingerprint") != fingerprint:
            raise ValueError("--resume refused: temporary run input fingerprint differs")
    else:
        temporary.mkdir(parents=True)
        metadata = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "system_count": len(records),
            "input_fingerprint": fingerprint,
            "excluded_paths": [str(data_root / "trash")],
            "warnings": [
                "One 10 ps trajectory per condition; no between-condition significance tests.",
                "Solution-only pre-adsorption simulation; no substrate or direct SAM-quality observable.",
                "Duplicate CSV hashes are collapsed as one independent trajectory.",
            ],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    checkpoint_root = temporary / ".checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    opa_path = Path(__file__).resolve().parents[2] / "structures" / "opa.vasp"
    for index, record in enumerate(records, start=1):
        checkpoint = checkpoint_root / f"{record.group}__{record.system}"
        if (checkpoint / "metrics.json").exists():
            print(f"[{index:02d}/{len(records)}] resumed {record.group}/{record.system}", flush=True)
            continue
        print(f"[{index:02d}/{len(records)}] analyzing {record.group}/{record.system}", flush=True)
        core = compute_core_metrics(record)
        with Trajectory(str(record.traj_path), mode="r") as trajectory:
            structural_frames = len(trajectory)
        if structural_frames == record.n_frames:
            solvation, profiles = compute_solvation_metrics(record, opa_path)
            solvation["structural_data_status"] = "matched"
        else:
            solvation = {
                "group": record.group,
                "system": record.system,
                "n_trajectory_frames": structural_frames,
                "structural_data_status": (
                    f"excluded: trajectory has {structural_frames} frames but CSV has "
                    f"{record.n_frames}; sources diverge after frame 0"
                ),
                **{column: float("nan") for column in SOLVATION_BASE_COLUMNS},
            }
            profiles = pd.DataFrame(columns=PROFILE_COLUMNS)
        system_metrics = {**core, **solvation, **asdict(record)}
        for key in ("csv_path", "traj_path", "xyz_path"):
            system_metrics[key] = Path(system_metrics[key]).as_posix()
        checkpoint.mkdir(parents=True)
        (checkpoint / "metrics.json").write_text(
            json.dumps(system_metrics, ensure_ascii=False, allow_nan=True), encoding="utf-8"
        )
        pd.DataFrame(block_sensitivity(record)).to_csv(checkpoint / "blocks.csv", index=False)
        profiles.to_csv(checkpoint / "profiles.csv", index=False)

    metrics, blocks, profiles = _collect_checkpoints(checkpoint_root)
    order = {(record.group, record.system): index for index, record in enumerate(records)}
    metrics["_order"] = [order[(group, system)] for group, system in zip(metrics.group, metrics.system)]
    metrics = metrics.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    metrics = collapse_duplicate_trajectories(metrics)
    matched = metrics["n_trajectory_frames"] == metrics["n_frames"]
    metrics.loc[metrics["structural_data_status"].isna() & matched, "structural_data_status"] = "matched"
    metrics.loc[metrics["structural_data_status"].isna() & ~matched, "structural_data_status"] = "unverified mismatch"
    metrics["solvent_class"] = metrics["base_solvent"].map(SOLVENT_CLASS).fillna("mixture")

    write_manifest(records, temporary / "manifest.csv")
    _write_frame(metrics, temporary / "tables" / "system_metrics.csv")
    _write_frame(blocks, temporary / "tables" / "block_metrics.csv")
    _write_frame(profiles, temporary / "tables" / "rdf_profiles.csv")
    for group in ("group1", "group2", "group3"):
        _write_frame(
            metrics.loc[metrics["group"] == group], temporary / "groups" / f"{group}_metrics.csv"
        )
    _write_comparisons(metrics, temporary / "comparisons")

    literature_root = temporary / "literature"
    literature_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(literature_path, literature_root / "sams_solvent_methods.json")
    literature_frame = pd.DataFrame(literature)
    _write_frame(literature_frame, literature_root / "literature_matrix.csv")

    generated = render_group_reports(metrics, profiles, blocks, temporary)
    index_paths = render_groupwise_index(metrics, temporary)
    generated.extend(index_paths)
    group_report_paths = [
        path for path in generated
        if path.parent.name == "reports" and path.name != "index.md" and path.name != "index.html"
    ]
    report_paths = [*index_paths, *group_report_paths]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mismatched = metrics.loc[metrics["structural_data_status"] != "matched"]
    if not mismatched.empty:
        metadata["warnings"].extend(
            f"{row.group}/{row.system}: {row.structural_data_status}"
            for row in mismatched.itertuples()
        )
    metadata.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "git_commit": _git_commit(),
            "parameters": {
                "rdf_range_A": [0.0, 12.0],
                "rdf_bin_A": 0.1,
                "coordination_cutoff_A": 4.0,
                "tamsd_max_lag_fraction": 1 / 3,
            },
            "input_hashes": {f"{r.group}/{r.system}": r.sha256 for r in records},
            "duplicate_trajectory_count": int(metrics["is_duplicate_copy"].sum()),
            "report_paths": [str(path.relative_to(temporary)).replace("\\", "/") for path in report_paths],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(checkpoint_root)
    outputs = sorted(
        str(path.relative_to(temporary)).replace("\\", "/")
        for path in temporary.rglob("*") if path.is_file()
    )
    metadata["outputs"] = outputs
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output_root)
    final_reports = tuple(output_root / path.relative_to(temporary) for path in report_paths)
    return RunSummary(len(records), 0, output_root, final_reports, "complete")
