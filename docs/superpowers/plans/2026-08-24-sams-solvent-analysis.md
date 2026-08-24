# SAMs Solvent Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible Python workflow that analyzes the three corrected OPA solvent datasets, adds mixture-specific preferential-solvation metrics, verifies high-impact literature methods, and produces a concise Chinese report without overwriting historical results.

**Architecture:** Add a focused `analysis/new_solvent_analysis/` package with separate modules for manifest construction, trajectory/CSV metrics, species-resolved solvation, group comparisons, and reporting. A thin runner orchestrates these modules and writes only to `results/new_solvent_analysis_2026-08-24/`; unit tests use small synthetic arrays/trajectories, while one integration test exercises the real input manifest without loading all trajectories.

**Tech Stack:** Python 3.10, NumPy, pandas, matplotlib, ASE, scikit-learn compatibility with the existing scripts, standard-library `unittest`, JSON/CSV/Markdown/HTML outputs.

## Global Constraints

- Treat `data/group1`, `data/group2`, and `data/group3` as the corrected source of truth.
- Exclude every directory under `data/trash`.
- Do not overwrite existing historical files under `results/` or modify raw files under `data/`.
- Write new outputs only to `results/new_solvent_analysis_2026-08-24/`.
- Interpret the simulations as single-OPA pre-adsorption solvation, not direct SAM growth or monolayer quality.
- Treat identical copied datasets as one trajectory, not independent replicates.
- Do not report classical between-condition significance tests because each condition has one trajectory.
- Keep the existing user changes in the dirty worktree intact.

---

## File Structure

- `requirements-analysis.txt`: reproducible local analysis dependencies.
- `analysis/new_solvent_analysis/__init__.py`: package version and public entry point.
- `analysis/new_solvent_analysis/manifest.py`: group definitions, filename resolution, checksums, composition metadata, and validation.
- `analysis/new_solvent_analysis/core_metrics.py`: CSV metrics, COM unwrapping, TAMSD, alpha fits, and block sensitivity.
- `analysis/new_solvent_analysis/solvation.py`: OPA head/tail topology, total and species-resolved RDF/CN, and enrichment factors.
- `analysis/new_solvent_analysis/comparisons.py`: group-specific tables, Pearson/Spearman trends, leave-one-out sensitivity, and duplicate handling.
- `analysis/new_solvent_analysis/reporting.py`: publication-style plots plus Markdown/HTML assembly.
- `analysis/new_solvent_analysis/pipeline.py`: deterministic end-to-end orchestration and run metadata.
- `analysis/run_new_solvent_analysis.py`: command-line runner.
- `analysis/literature/sams_solvent_methods.json`: verified literature evidence bank with DOI/URL/citation-count provenance.
- `tests/test_new_manifest.py`, `tests/test_new_core_metrics.py`, `tests/test_new_solvation.py`, `tests/test_new_comparisons.py`, `tests/test_new_pipeline.py`: unit and integration tests.

---

### Task 1: Reproducible environment and validated input manifest

**Files:**
- Create: `requirements-analysis.txt`
- Create: `analysis/new_solvent_analysis/__init__.py`
- Create: `analysis/new_solvent_analysis/manifest.py`
- Create: `tests/test_new_manifest.py`

**Interfaces:**
- Produces: `SystemRecord` dataclass; `build_manifest(data_root: Path) -> list[SystemRecord]`; `write_manifest(records: list[SystemRecord], path: Path) -> None`.
- `SystemRecord` fields: `group`, `system`, `role`, `csv_path`, `traj_path`, `xyz_path`, `file_prefix`, `cosolvent`, `base_solvent`, `cosolvent_fraction`, `n_frames`, `time_start_ps`, `time_end_ps`, `sha256`.

- [ ] **Step 1: Create the failing manifest tests**

```python
class ManifestTests(unittest.TestCase):
    def test_real_manifest_has_27_valid_systems_and_no_trash(self):
        records = build_manifest(Path("data"))
        self.assertEqual(len(records), 27)
        self.assertEqual(Counter(r.group for r in records), {"group1": 7, "group2": 13, "group3": 7})
        self.assertFalse(any("trash" in str(r.csv_path) for r in records))

    def test_acetone_five_percent_resolves_nonmatching_file_prefix(self):
        record = next(r for r in build_manifest(Path("data"))
                      if r.group == "group1" and r.system == "acetone_5_n-heptane_95")
        self.assertEqual(record.file_prefix, "acetone_n-heptane")
        self.assertEqual(record.cosolvent_fraction, 0.05)

    def test_group3_roles_are_stratified(self):
        roles = {r.system: r.role for r in build_manifest(Path("data")) if r.group == "group3"}
        self.assertEqual(roles["thf_n-heptane"], "n-heptane-base-composition")
        self.assertEqual(roles["thf_toluene"], "alternate-base-composition")
        self.assertEqual(roles["ethyl_acetate"], "pure-reference")
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_new_manifest -v`

Expected: `ModuleNotFoundError: analysis.new_solvent_analysis`.

- [ ] **Step 3: Create the workspace environment and install exact project requirements**

Create `requirements-analysis.txt` with:

```text
-r requirements.txt
scikit-learn
```

Run:

```powershell
uv venv .venv --python C:\cygwin_wm\opt_win\miniconda3\python.exe
uv pip install --python .\.venv\Scripts\python.exe -r requirements-analysis.txt
```

- [ ] **Step 4: Implement manifest discovery and strict validation**

Implement deterministic group ordering, glob-based prefix resolution, CSV header/frame/time validation, required-file checks, explicit group/role metadata, and SHA-256 hashing of the motion/force CSV. Raise `ManifestError` with the system path and failed invariant instead of silently skipping data.

- [ ] **Step 5: Run manifest tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_new_manifest -v`

Expected: 3 tests pass and the manifest reports 7/13/7 systems.

- [ ] **Step 6: Commit the manifest unit**

```powershell
git add requirements-analysis.txt analysis/new_solvent_analysis/__init__.py analysis/new_solvent_analysis/manifest.py tests/test_new_manifest.py
git commit -m "feat: validate new solvent analysis inputs"
```

---

### Task 2: Core motion, force, TAMSD, and robustness metrics

**Files:**
- Create: `analysis/new_solvent_analysis/core_metrics.py`
- Create: `tests/test_new_core_metrics.py`

**Interfaces:**
- Consumes: `SystemRecord` from Task 1.
- Produces: `unwrap_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray`; `tamsd(positions: np.ndarray, max_lag: int) -> np.ndarray`; `fit_anomalous_exponent(times: np.ndarray, msd: np.ndarray, min_lag: int = 2) -> AlphaFit`; `compute_core_metrics(record: SystemRecord) -> dict[str, float | str]`; `block_sensitivity(record: SystemRecord, blocks: int = 4) -> list[dict[str, float]]`.

- [ ] **Step 1: Write failing numerical tests**

```python
class CoreMetricTests(unittest.TestCase):
    def test_unwrap_removes_periodic_jump(self):
        wrapped = np.array([[9.5, 0, 0], [0.2, 0, 0], [0.9, 0, 0]])
        got = unwrap_positions(wrapped, np.diag([10.0, 10.0, 10.0]))
        np.testing.assert_allclose(got[:, 0], [9.5, 10.2, 10.9])

    def test_tamsd_matches_linear_track(self):
        pos = np.arange(6.0)[:, None] * np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(tamsd(pos, 3), [1.0, 4.0, 9.0])

    def test_alpha_for_ballistic_track_is_two(self):
        tau = np.arange(1.0, 11.0)
        fit = fit_anomalous_exponent(tau, tau ** 2)
        self.assertAlmostEqual(fit.alpha, 2.0, places=10)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=10)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_new_core_metrics -v`

Expected: import failure for `core_metrics`.

- [ ] **Step 3: Implement the minimal numerical functions**

Use minimum-image fractional-coordinate unwrapping for periodic cells. Calculate TAMSD with all valid time origins. Fit `log(MSD) = alpha * log(tau) + intercept`, return alpha, R², fit bounds, and the descriptive `D_alpha` prefactor.

- [ ] **Step 4: Implement per-system observable and block metrics**

Read the CSV by actual resolved prefix. Compute final/mean/max displacement; force mean/standard deviation/maximum; force-displacement Pearson correlation; TAMSD alpha; and four contiguous-block values for mean force, RMS displacement from block origin, and alpha where the block length permits fitting. Preserve historical single-origin `D_eff` only under the explicit name `legacy_single_origin_d_eff`.

- [ ] **Step 5: Run numerical tests and one real-system smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_new_core_metrics -v
.\.venv\Scripts\python.exe -c "from pathlib import Path; from analysis.new_solvent_analysis.manifest import build_manifest; from analysis.new_solvent_analysis.core_metrics import compute_core_metrics; r=build_manifest(Path('data'))[0]; m=compute_core_metrics(r); assert m['n_frames']==101 and m['alpha_fit_r2']<=1.0; print(r.system, m['alpha'])"
```

Expected: tests pass; smoke test prints one system and a finite alpha.

- [ ] **Step 6: Commit the core-metrics unit**

```powershell
git add analysis/new_solvent_analysis/core_metrics.py tests/test_new_core_metrics.py
git commit -m "feat: add robust OPA motion and force metrics"
```

---

### Task 3: Total and species-resolved solvation structure

**Files:**
- Create: `analysis/new_solvent_analysis/solvation.py`
- Create: `tests/test_new_solvation.py`

**Interfaces:**
- Consumes: `SystemRecord` and ASE trajectory paths.
- Produces: `classify_opa_head_tail(opa_path: Path) -> OpaTopology`; `assign_species_blocks(symbols: Sequence[str], templates: dict[str, tuple[str, ...]], molecule_counts: dict[str, int]) -> dict[str, list[int]]`; `assign_species(record: SystemRecord, symbols: Sequence[str]) -> SpeciesAssignment`; `radial_profile(...) -> RadialProfile`; `compute_solvation_metrics(record: SystemRecord, opa_path: Path) -> tuple[dict, pd.DataFrame]`.

- [ ] **Step 1: Write failing topology, species, and enrichment tests**

```python
class SolvationTests(unittest.TestCase):
    def test_enrichment_is_one_when_local_matches_bulk(self):
        self.assertAlmostEqual(enrichment_factor(local_count=5, local_total=20,
                                                 bulk_fraction=0.25), 1.0)

    def test_species_assignment_consumes_every_atom_once(self):
        templates = {"acetone": ("C", "C", "C", "O"),
                     "n-heptane": ("C", "C", "H", "H")}
        symbols = templates["acetone"] * 2 + templates["n-heptane"] * 3
        blocks = assign_species_blocks(symbols, templates,
                                       {"acetone": 2, "n-heptane": 3})
        self.assertEqual(sum(len(v) for v in blocks.values()), len(symbols))
        self.assertEqual(set(blocks), {"acetone", "n-heptane"})

    def test_minimum_image_distance(self):
        d = minimum_image(np.array([9.0, 0.0, 0.0]), np.diag([10.0, 10.0, 10.0]))
        np.testing.assert_allclose(d, [-1.0, 0.0, 0.0])
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_new_solvation -v`

Expected: import failure for `solvation`.

- [ ] **Step 3: Implement OPA topology and deterministic species assignment**

Reuse the chemically defined phosphonate head from `structures/opa.vasp` (P + three O + two acidic H) and the terminal C18 carbon. Validate that OPA has 61 atoms and occurs first in every trajectory. Derive molecule blocks from known component structure symbol sequences and manifest composition; reject any assignment that does not consume every solvent atom exactly once.

- [ ] **Step 4: Implement RDF, coordination number, and preferential-solvation calculations**

Calculate minimum-image distances for all frames, normalized RDFs on a fixed 0–12 Å grid with 0.1 Å bins, running coordination numbers, CN at 4 Å, first-peak position/height, and species-local fractions. Define enrichment as `(local species fraction)/(bulk molecular fraction)` and report numerator, denominator, and radius alongside the ratio.

- [ ] **Step 5: Run solvation tests and one trajectory smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_new_solvation -v
.\.venv\Scripts\python.exe -c "from pathlib import Path; from analysis.new_solvent_analysis.manifest import build_manifest; from analysis.new_solvent_analysis.solvation import compute_solvation_metrics; r=next(x for x in build_manifest(Path('data')) if x.system=='acetone_1_n-heptane_99'); m,p=compute_solvation_metrics(r,Path('structures/opa.vasp')); assert len(p)>0 and m['head_coord_number_4A']>=0; print(m['head_coord_number_4A'])"
```

Expected: tests pass and the real trajectory returns a finite non-negative head CN.

- [ ] **Step 6: Commit the solvation unit**

```powershell
git add analysis/new_solvent_analysis/solvation.py tests/test_new_solvation.py
git commit -m "feat: resolve preferential OPA solvation by species"
```

---

### Task 4: Group comparisons and small-sample sensitivity

**Files:**
- Create: `analysis/new_solvent_analysis/comparisons.py`
- Create: `tests/test_new_comparisons.py`

**Interfaces:**
- Consumes: merged per-system metric DataFrame and manifest roles.
- Produces: `pearson_spearman(x, y) -> CorrelationPair`; `leave_one_out_correlations(x, y) -> list[float]`; `build_group1_comparison(df) -> ComparisonResult`; `build_group2_comparison(df) -> ComparisonResult`; `build_group3_comparison(df) -> ComparisonResult`.

- [ ] **Step 1: Write failing comparison tests**

```python
class ComparisonTests(unittest.TestCase):
    def test_monotonic_series_has_unit_correlations(self):
        corr = pearson_spearman(np.arange(5.0), np.arange(5.0) ** 3)
        self.assertGreater(corr.pearson, 0.94)
        self.assertEqual(corr.spearman, 1.0)

    def test_group3_core_excludes_pure_and_alternate_base_rows(self):
        frame = pd.DataFrame({
            "system": ["acetone_5_n-heptane_95", "isopropanol_5_n-heptane_95",
                       "thf_n-heptane", "toluene_n-heptane", "thf_toluene",
                       "ethyl_acetate", "propylene_carbonate"],
            "role": ["n-heptane-base-composition"] * 4
                    + ["alternate-base-composition", "pure-reference", "pure-reference"],
            "alpha": np.arange(7.0),
        })
        result = build_group3_comparison(frame)
        self.assertEqual(set(result.core_rows.system),
                         {"acetone_5_n-heptane_95", "isopropanol_5_n-heptane_95",
                          "thf_n-heptane", "toluene_n-heptane"})

    def test_duplicate_checksum_is_not_counted_as_replicate(self):
        frame = pd.DataFrame({"system": ["copy_a", "copy_b"],
                              "sha256": ["same-hash", "same-hash"]})
        result = collapse_duplicate_trajectories(frame)
        self.assertEqual(result.independent_trajectory.nunique(), 1)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_new_comparisons -v`

Expected: import failure for `comparisons`.

- [ ] **Step 3: Implement rank correlations, leave-one-out ranges, and group stratification**

Implement average ranks for ties without SciPy, Pearson on raw values, Spearman on ranks, and leave-one-out correlation ranges. Group 1 sorts by acetone fraction and flags direction changes. Group 2 reports solvent and manual chemistry-class descriptive summaries. Group 3 exposes separate `core_rows`, `alternate_base_rows`, and `pure_reference_rows`; only the four n-heptane-base rows enter the core ranking.

- [ ] **Step 4: Run all comparison tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_new_comparisons -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the comparison unit**

```powershell
git add analysis/new_solvent_analysis/comparisons.py tests/test_new_comparisons.py
git commit -m "feat: compare solvent groups with small-sample safeguards"
```

---

### Task 5: Verified literature evidence bank

**Files:**
- Create: `analysis/literature/sams_solvent_methods.json`

**Interfaces:**
- Produces a JSON array whose records contain: `title`, `authors`, `year`, `journal`, `doi`, `url`, `citation_count`, `citation_count_source`, `citation_count_checked_on`, `system`, `method_used`, `metric_reused_here`, `applicability`, and `limitations`.

- [ ] **Step 1: Search primary bibliographic sources**

Search for phosphonic-acid and alkanethiol SAM solvent effects, solvent-controlled assembly kinetics/order, molecular-dynamics RDF/coordination analysis, preferential solvation, and short-trajectory TAMSD fitting. Prefer publisher pages, DOI records, PubMed, Crossref, and OpenAlex; use citation counts only when the source and query date are recorded.

- [ ] **Step 2: Verify each retained reference independently**

For every record, confirm title, journal, year, DOI, and that the paper actually uses or justifies the method attributed to it. Exclude unverifiable references and papers that discuss SAM solvents without an applicable analysis method.

- [ ] **Step 3: Write the evidence bank**

Include at least six verified sources spanning: direct SAM solvent comparison, phosphonic-acid monolayer simulation/experiment, RDF or coordination analysis, and anomalous-diffusion/TAMSD methodology. Quote no more than 25 words from any source; store paraphrased method notes.

- [ ] **Step 4: Validate JSON and DOI uniqueness**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import json,pathlib; p=pathlib.Path('analysis/literature/sams_solvent_methods.json'); x=json.loads(p.read_text(encoding='utf-8')); assert len(x)>=6; dois=[r['doi'].lower() for r in x]; assert len(dois)==len(set(dois)); assert all(r['url'] and r['method_used'] and r['citation_count_checked_on']=='2026-08-24' for r in x); print(len(x))"
```

Expected: prints a source count of at least 6.

- [ ] **Step 5: Commit the evidence bank**

```powershell
git add analysis/literature/sams_solvent_methods.json
git commit -m "docs: verify SAM solvent analysis literature"
```

---

### Task 6: Figures and report generation

**Files:**
- Create: `analysis/new_solvent_analysis/reporting.py`
- Create: `tests/test_new_pipeline.py`

**Interfaces:**
- Consumes: validated manifest, metrics, RDF tables, comparison results, and literature JSON.
- Produces: `render_group_reports(...) -> list[Path]`; `render_combined_report(...) -> list[Path]`; figure files with stable names and a report link map.

- [ ] **Step 1: Write failing report-output tests**

```python
class ReportingTests(unittest.TestCase):
    def test_report_contains_scope_and_uncertainty_language(self):
        context = {"system_count": 27, "headline_findings": [],
                   "limitations": ["每个条件只有单条轨迹", "模拟描述预吸附溶剂化"]}
        text = build_combined_markdown(context)
        self.assertIn("预吸附", text)
        self.assertIn("单条轨迹", text)
        self.assertNotIn("显著提高 SAM 生长", text)

    def test_all_figure_labels_are_present_in_manifest(self):
        frame = pd.DataFrame({"system": ["acetone_1_n-heptane_99",
                                         "acetone_5_n-heptane_95"],
                              "cosolvent_fraction": [0.01, 0.05],
                              "alpha": [0.8, 0.9],
                              "mean_force_eV_A": [0.4, 0.5]})
        paths = render_group1_figures(frame, self.tmpdir)
        self.assertTrue(all(p.exists() and p.stat().st_size > 0 for p in paths))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_new_pipeline.ReportingTests -v`

Expected: import failure for `reporting`.

- [ ] **Step 3: Implement a consistent scientific figure system**

Use a colorblind-safe palette, explicit units, 300 dpi PNG plus PDF/SVG where supported, no 3D charts, and no bar chart that implies uncertainty unsupported by replicates. Group 1 gets concentration-response and window-sensitivity panels; Group 2 gets ordered dot/range plots and chemistry-class overlays; Group 3 gets the four-condition core comparison plus separate reference panels; combined output gets method-linked structure/dynamics relationships.

- [ ] **Step 4: Implement Markdown and HTML reports**

Each report must include inputs, method, results, direct-evidence findings, literature-informed interpretations, uncertainty, and file provenance. Generate HTML from the same structured context as Markdown so numerical values cannot diverge between formats.

- [ ] **Step 5: Run report tests and inspect generated test figures**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_new_pipeline.ReportingTests -v`

Expected: all report tests pass and every generated image is non-empty.

- [ ] **Step 6: Commit the reporting unit**

```powershell
git add analysis/new_solvent_analysis/reporting.py tests/test_new_pipeline.py
git commit -m "feat: render SAM solvent analysis reports"
```

---

### Task 7: End-to-end pipeline and real-data execution

**Files:**
- Create: `analysis/new_solvent_analysis/pipeline.py`
- Create: `analysis/run_new_solvent_analysis.py`
- Modify: `tests/test_new_pipeline.py`

**Interfaces:**
- Consumes all Task 1–6 interfaces.
- Produces: `run_analysis(data_root: Path, output_root: Path, literature_path: Path, dry_run: bool = False, resume: bool = False) -> RunSummary`; CLI flags `--data-root`, `--output-root`, `--literature`, and `--resume`.

- [ ] **Step 1: Add a failing dry-run integration test**

```python
class PipelineTests(unittest.TestCase):
    def test_dry_run_validates_all_inputs_without_writing_results(self):
        summary = run_analysis(Path("data"), self.tmpdir / "out",
                               Path("analysis/literature/sams_solvent_methods.json"),
                               dry_run=True)
        self.assertEqual(summary.system_count, 27)
        self.assertFalse((self.tmpdir / "out").exists())
```

- [ ] **Step 2: Run the integration test and verify failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_new_pipeline.PipelineTests -v`

Expected: import or signature failure for `pipeline.run_analysis`.

- [ ] **Step 3: Implement atomic output orchestration**

Write the run into `results/.new_solvent_analysis_2026-08-24.tmp/`, validate all expected artifacts, then rename to `results/new_solvent_analysis_2026-08-24/`. Refuse to replace an existing completed directory; accept `--resume` only for a matching `analysis_run.json` input manifest hash.

- [ ] **Step 4: Add full-run metadata**

Write `analysis_run.json` with Python/package versions, Git commit, input hashes, parameters, start/end timestamps, excluded paths, warnings, and every output relative path. Write `manifest.csv`, per-system metrics, RDF tables, group comparison tables, literature matrix, figures, Markdown reports, and HTML reports.

- [ ] **Step 5: Run the full unit suite**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_new_*.py" -v`

Expected: all tests pass.

- [ ] **Step 6: Run the real 27-system analysis**

Run:

```powershell
.\.venv\Scripts\python.exe analysis\run_new_solvent_analysis.py --data-root data --output-root results\new_solvent_analysis_2026-08-24 --literature analysis\literature\sams_solvent_methods.json
```

Expected: exits 0 and prints 27 analyzed systems, 0 trash systems, and paths for three group reports plus the combined report.

- [ ] **Step 7: Commit code without committing ignored/generated results**

```powershell
git add analysis/new_solvent_analysis/pipeline.py analysis/run_new_solvent_analysis.py tests/test_new_pipeline.py
git commit -m "feat: run three-group SAM solvent analysis"
```

---

### Task 8: Verification, scientific audit, and final handoff

**Files:**
- Verify: `results/new_solvent_analysis_2026-08-24/**`
- Modify only if audit finds a defect: the responsible module and its focused test.

**Interfaces:**
- Consumes: completed result tree and source data.
- Produces: verified final reports and a concise user-facing Chinese summary.

- [ ] **Step 1: Verify artifact completeness and numerical finiteness**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import pandas as pd,json; r=Path('results/new_solvent_analysis_2026-08-24'); m=pd.read_csv(r/'manifest.csv'); assert len(m)==27 and not m.astype(str).apply(lambda c:c.str.contains('trash')).any().any(); s=pd.read_csv(r/'combined/system_metrics.csv'); assert len(s)==27; meta=json.loads((r/'analysis_run.json').read_text(encoding='utf-8')); assert meta['status']=='complete'; print('verified',len(s))"
```

Expected: `verified 27`.

- [ ] **Step 2: Recompute selected values independently**

Select one Group 1, one Group 2, and one Group 3 system. Independently recompute CSV force mean/final displacement and compare to `system_metrics.csv` with relative tolerance `1e-10`. Independently verify one 4 Å species count against the saved radial profile.

- [ ] **Step 3: Visually inspect every figure family**

Open at least one concentration-response figure, one pure-solvent comparison, one species-resolved RDF/enrichment figure, and the combined overview. Check clipping, labels, units, legends, ordering, and color distinction; fix the responsible renderer and add a regression assertion for any defect.

- [ ] **Step 4: Audit claims against evidence and limitations**

For each headline conclusion, label it as direct data, inference, or recommendation. Confirm that non-monotonic concentration behavior, correlation claims, and solvent-class explanations disclose single-trajectory and short-duration limitations. Confirm every literature-supported statement links to a verified source.

- [ ] **Step 5: Re-run verification after any corrections**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_new_*.py" -v
.\.venv\Scripts\python.exe analysis\run_new_solvent_analysis.py --data-root data --output-root results\new_solvent_analysis_2026-08-24 --literature analysis\literature\sams_solvent_methods.json --resume
```

Expected: all tests pass and the run metadata remains `status: complete`.

- [ ] **Step 6: Deliver the concise Chinese report**

Report the strongest Group 1 concentration trend, Group 2 solvent ranking/class pattern, Group 3 preferential-solvation comparison, the most important methodological literature connection, and the limitations. Link the combined report, three group reports, metrics CSV, literature matrix, and analysis script.
