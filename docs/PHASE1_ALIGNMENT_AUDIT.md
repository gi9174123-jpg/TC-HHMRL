# Phase 1 Alignment Audit

## Scope
This audit records the state of the repository after phase-1 alignment to the teacher-revised mathematical model. The goal of this phase was code-truth alignment only. No formal experiment reruns or paper-figure regeneration are included here.

## Alignment Status

### 1. Site-specific episode task: aligned
- A single `TaskSpec` contract now defines the task truth fields:
  - `site_id`
  - `task_source`
  - `distances = (d0, d1, d2)`
  - `attenuation_c`
  - `misalign_std`
  - `amb_temp`
  - `gamma`
  - `delta`
  - `qos_min_rate`
  - `alignment_version`
  - `task_summary_version`
- `TaskSampler` supports `site_bank` and keeps `global_fallback` for compatibility only.
- `env.reset(options=...)` accepts `site_id`, `distances`, and task overrides.
- `_task_dict()` exposes:
  - `site_id`
  - `distance_tx0/1/2`
  - `task_source`
  - alignment metadata

### 2. HY semantics: aligned
- HY mode now uses the teacher-model energy-conserving split:
  - `info_share = tau * (1 - rho)`
  - `eh_share = 1 - tau * (1 - rho)`
- PS and TS semantics remain unchanged.
- The environment now asserts HY consistency locally.
- The downstream chain is covered by tests:
  - `qos_rate`
  - `eh_metric`
  - reward terms
  - info dict fields

### 3. Context task summary: aligned to `site_v2`
- The auxiliary task summary is now 9D:
  - `attenuation_c`
  - `misalign_std`
  - `amb_temp`
  - `gamma`
  - `delta`
  - `qos_min_rate`
  - `distance_tx0`
  - `distance_tx1`
  - `distance_tx2`
- `HierarchicalAgent.context_task_dim = 9`
- Train and eval transitions both propagate the 9D summary.

### 4. Version guard metadata: aligned
New runs now emit:
- `alignment_version = "system_model_v1"`
- `task_summary_version = "site_v2"`
- `pre_alignment = false`
- `task_source = "site_bank" | "global_fallback"`

Old checkpoints loaded without alignment metadata are treated as:
- `pre_alignment = true`

This guard is present in:
- environment/task info
- run summaries
- scenario summaries
- top-level benchmark report
- checkpoint metadata for the main hierarchical agent and SAC-Lagrangian baseline

### 5. Task summary builder and task hashes: aligned
- `build_task_summary_v2(...)` is now the single builder used for the 9D auxiliary task summary.
- Fixed task batches now emit stable SHA256 hashes for:
  - selection tasks
  - eval tasks
  - env-realism tasks
- These hashes are written into run summaries and scenario summaries.

## Audit Findings

### Raw/executed semantics: intact
- Replay still stores both raw and executed actions.
- Lower-level learning still uses executed action semantics.
- Upper-level target semantics remain executed-macro based.
- Phase-1 changes did not modify these invariants.

### Safety projection semantics: still soft-protection
- The current safety layer still implements predictive smooth derating in the main path.
- Phase-1 did not convert thermal protection into a hard constraint layer.
- This remains consistent with the intended “soft protection” wording for the method section.

### Fixed-task benchmark sharing: intact
- Fixed-task benchmark sampling remains shared across variants.
- Site-aware tasks are now the unit being shared, rather than only legacy environment ranges.
- Regression tests cover task sharing across variants.

### Curriculum and balanced sampling: no break found, but keep one restriction
- `TaskSampler.sample_tasks()` now balances across `site_bank` when `balanced_sampling = true`.
- `MetaTrainer._sampler_cfg_for_iter()` merges the curriculum phase sampler config on top of the base sampler config.
- This preserves `site_bank` as long as curriculum phases do **not** override `site_bank` themselves.

Restriction for phase 2:
- Do not define curriculum phases that replace `site_bank` unless this is intentional and separately audited.

### Site-bank validation: implemented
The repository now checks:
- unique `site_id`
- `distances` length is 3
- each range uses `lo <= hi`
- sites are not all identical across ranges and distances
- `strict_site_bank = true` now rejects missing or invalid `site_bank`

### Formal comparability filter: partially enforced in code
- New benchmark aggregation now rejects non-comparable run-summary rows at load time.
- Old checkpoints may still be loaded for compatibility, but they are marked `pre_alignment = true`.
- The main hierarchical agent and SAC-Lagrangian baseline both expose alignment metadata that supports this distinction.

## Known Gaps Left for Phase 2

### 1. Figure/report consumers outside the benchmark harness are not yet all filtered
The core benchmark aggregation path now rejects non-comparable run-summary rows, but external figure/report scripts still need to reject:
- runs missing `alignment_version`
- runs missing `task_summary_version`
- runs with `pre_alignment = true`

This is a deliberate split: phase 1 now writes the guard fields and enforces them in the benchmark harness; phase 2 must finish the same rule in all downstream figure/report consumers.

### 2. Thermal supplementary is not yet redesigned
Interfaces are now ready for a new thermal scenario, but the actual `thermal_rebalanced` supplementary setting is not part of phase 1.

### 3. External baselines are not yet implemented
Not included in this phase:
- `shin2024_matched`
- `dalal2018_safe`

## Paper Sync TODO
The paper must be updated to match `site_v2` semantics.

Required sync items:
1. The auxiliary task-summary notation must expand to the 9D site-aware form.
2. The text must reflect site-specific episode tasks instead of a purely range-sampled task abstraction.
3. The HY-mode equations must remain consistent with the energy-conserving implementation.
4. Any wording that implies a single compressed distance term should be revised to explicit per-source distances.

## Verification Performed
- Targeted regression tests after phase-1 modifications
- Full test suite execution

Result:
- `38 passed`
- no functional regressions detected in the current test suite
