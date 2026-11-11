# Manuscript analyses — one directory per result

Each figure/table has its own folder. Shared computations that generate several outputs are stored **once**, under Figure 4, and cross-referenced from other result folders to avoid divergences.

| Manuscript result | Code location | Status |
| --- | --- | --- |
| Figure 1: global cyclone track density | `Figure_1_Global_Cyclone_Track_Density/` | Original generator supplied |
| Table 1: cyclone counts | `Table_1_Cyclone_Counts_by_Season_and_Region/` | Original generator supplied; needs local storm-ID reference |
| Figure 4a: overall MAE | `Figure_04_Overall_Performance/Figure_04a_MAE/` | V9 generator |
| Figure 4b: pairwise KS / winning ratios | `Figure_04_Overall_Performance/Figure_04b_KS_Win/` | Standalone Nov 4 script + V9 |
| Figure 4c: IFS advantage KDE | `Figure_04_Overall_Performance/Figure_04c_IFS_Advantage/` | V9 generator |
| Figure 4d: track-length correlations | `Figure_04_Overall_Performance/Figure_04d_Pearson_Track_Length/` | Nov 4 script + Nov 8 updated intensity script and variance adjustment |
| Figure 5: MSL-only lookback | `Figure_05_Lookback/` | Source script not yet provided |
| Figure 6: intensity-change MAE | `Figure_06_Intensity_Change/` | V9 candidate, verify final |
| Figure 7: domain generalization MAE vs lead | `Figure_07_Domain_Generalization/` | Script + manuscript PDF (Nov 11) |
| Table 5: 2010 model metrics | `Table_05_Performance_Metrics/` | V9 generator |
| 2010 IFS audits | `IFS_2010_Verification/` | Supporting code |
| Supplementary: full-dataset metrics + pairwise KS (combined) | `Supplementary_Table_Full_Dataset_Metrics_KS/` | Metrics recomputed and audited on real data (Nov 11); KS section not yet recomputed |
| Figure S1: IFS vs ERA5 local minimum | `Figure_S1_IFS_ERA5_Local_Minimum/` | Nov 9 four-panel script, storm `JJA_2010_NH_neg_135` |

## Local data and outputs

All input atmospheric arrays, model predictions, checkpoint artifacts, and full-data CSVs stay outside source control. The repository's `.gitignore` excludes these types, and the package does not include any raw data. Use locally available `data/` inputs and write results under `outputs/` (also ignored).

## Reproducibility notes

Each README states which source was available and how to run it. Scripts are **syntax-checked** for this package but most have **not been end-to-end rerun** against the large original forecast datasets here. Do not assume that different figures have the same verification cohort or the same spatial truth. Historical `Figure5...` output filenames in V9 correspond to current **Figure 4**.
