# Supplementary table — Full-dataset (1979–2020) performance metrics and pairwise KS / winning ratios

One combined supplementary table for 6, 12, 18 and 24 h, replacing the former separate S2/S3 tables (6/12 h and 18/24 h) and the separate pairwise KS table, as requested by the supervisor.

## Files

| File | Purpose |
| --- | --- |
| `build_supplementary_full_dataset_table.py` | Recomputes MAE, Bias (+95% CI), STD and SS (+95% CI) from the original forecast CSVs and audits them against the previously published values. Writes LaTeX and audit CSVs. |
| `Supplementary_Metrics_and_KS_Combined.tex` | **Current manuscript table**: performance metrics plus pairwise winning ratio / KS distance, in one float. Carries both labels `tab:supp_full_metrics` and `tab:supp_ks_all_leads`. |
| `Supplementary_Combined_Full_Dataset_Metrics_FINAL_VERIFIED.tex` | Performance-metrics-only version (before adding the KS section). |
| `reference/FullDataset_Comparison_with_old_table.csv` | Audit output from the real-data run: old vs recomputed unrounded values and status per entry. |
| `reference/Supplementary_Full_Dataset_Validation_Summary.txt` | Summary of that audit. |

## Run

From the repository root:

```bash
python analysis/Supplementary_Table_Full_Dataset_Metrics_KS/build_supplementary_full_dataset_table.py \
  --base-dir data/updated_data
```

Expected inputs under `--base-dir` (not committed):

```
control/control_31.csv ... control_34.csv
leakage/leakage_6d.csv ... leakage_24d.csv
msl_only/all_cyclone_data_full_31_version_156_157_158_159_6h.csv
simple_nn/all_cyclone_data_full_31_version_0_1_2_3.csv
```

Outputs go to `outputs/supplementary_full_dataset/` (override with `--output-dir`). Options: `--bootstraps` (default 1000), `--seed` (default 42), `--include-msl-ss` (show MSL-only 6 h SS instead of `--`).

## Definitions

- Signed error = prediction − observed tracked-center MSLP (hPa). STD uses `ddof=1`.
- `SS = 1 − RMSE(model) / RMSE(Persistence)` (RMSE-based, not MAE-based).
- **Cohorts differ by statistic.** MAE/Bias/STD use each model's own available full-year steps. SS for Control, Leakage and Linear uses the complete-case intersection of Control + Linear + Persistence + Leakage on `(storm_key, step_index)`. SS therefore cannot be reconstructed from the displayed Bias/STD.
- SS 95% CI: cyclone-cluster bootstrap (whole storms resampled). Bias 95% CI: normal approximation (`1.96 × SD/√N`), which does not account for within-storm dependence.
- Header `N` is the maximum available step count per lead, not the N of every cell.

## Verification status (11 Nov 2026)

- The script was run on the real full-year CSVs. Of 129 previously published values, 125 are unchanged at three decimals and 4 changed:
  - Simple NN 6 h SS `0.228 (0.225, 0.230)` → `0.231 (0.229, 0.234)`
  - Linear Regression 18 h SS upper CI `0.180` → `0.179`
- MSL-only 6 h SS = `0.513 (0.511, 0.515)` was computed but is shown as `--`, matching the earlier reporting scope.
- Bootstrap CI endpoints can differ in the last digit from the original (unseeded) run.
- **The pairwise winning-ratio / KS section was transcribed from the previous table and has not yet been recomputed from forecast-level data.** Its `p < 0.05` statement uses standard `ks_2samp` p-values, which do not correct for temporal dependence within cyclones. See `../Figure_04_Overall_Performance/Figure_04b_KS_Win/` for the KS code.
