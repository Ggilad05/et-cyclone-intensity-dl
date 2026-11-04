# Figure 4 — Overall performance (current manuscript numbering)

The original V9 analysis (`build_overall_performance_v9.py`) is preserved **without changing its scientific calculations**. It generates:

- Figure 4a: multi-lead MAE overview (`Figure5a_MAE_original_cohorts.*` in legacy output)
- Figure 4b: pair-specific KS/winning ratio (V9 output; see the dedicated standalone script in `Figure_04b_KS_Win/`)
- Figure 4c: Control / Persistence vs IFS error-advantage KDE (`Figure5c_2010_strict_advantage_vs_IFS.*`)
- Table 5: 2010 three-way matched error metrics and bootstrap intervals
- Figure 6: full-year MAE vs intensity-change candidate and supplementary diagnostics

**Figure 4d is a separate calculation**, in `Figure_04d_Pearson_Track_Length/`.

## Run

From repository root (paths are examples; replace with your actual local inputs):

```bash
python analysis/Figure_04_Overall_Performance/build_overall_performance_v9.py \
  --base-dir data/updated_data \
  --ifs-dir data/ifs_2010 \
  --ifs-history data/source_all_valid_times.csv \
  --out-dir outputs/overall_performance
```

Use `--check-inputs` first to audit model paths and overlap. Required input CSVs are deliberately **not** committed. The script requires the complete ERA5 local-minimum history for physically consistent IFS persistence. See `--help` for optional ablation paths and bootstrap settings.

## Scientific and version notes

- Main Table 5: exact 2010 `(storm_key, step_index)` three-way intersection, different verification target for IFS versus CNN.
- Figure 4a: original **mixed-cohort** descriptive overview (IFS 2010, other models all years).
- Figure 4b: pair-specific matching (with IFS: 2010; without IFS: all years).
- Figure 4c: IFS/Control or IFS/Persistence pairwise 2010 samples, independently of Leakage.
- Figure 6 is currently named `candidate` by the source script; do not treat as the final figure without matching the manuscript.
- File naming inside V9 still says `Figure5` because Figure 4 was previously Figure 5 in the paper.
- Standard two-sample KS p-values in V9 do not correct for within-storm temporal dependence.

The standalone Figure 4b script is provided for provenance, but different output scripts should not be silently mixed when assembling the final four-panel composite.
