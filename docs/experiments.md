# Experiment log and decisions

## 2026-10-30 — Regional storm-list audit

- Original regional storm-ID lists were retained under `classifications/original/`.
- A revised SH South Atlantic list is stored under `classifications/revised/`, correcting at least one doubled `.npy` suffix.
- A revised Southern Ocean list excludes storm IDs that also occur in the South Atlantic list, preserving the smaller South Atlantic study cohort.
- Earlier audit of the **original lists** found 4,896 storm IDs shared by SA and SO (out of 5,250 SA and 24,796 SO listed storm IDs).
- The revised list is intended to avoid category overlap. Verify the **effective** training/test membership by unique storm ID in the current loader.
- Four separate lead-time DG trainings were initiated; results and final population counts are pending validation.
- Table 1 and Figure 1 need regeneration using the final, documented population and classification rules.

### Open audit items

- [ ] Verify actual exclusion reason for storms absent from each lead's sample cohort.
- [ ] Confirm whether 1979–2020 is enforced in all paper figures/tables.
- [ ] Confirm whether geographical inclusion is defined by genesis position, track intersection, or another criterion.
- [ ] Confirm train/test storm-ID disjointness in regional DG.
- [ ] Regenerate region and hemisphere/year counts and track steps from the same population as Figure 1.
- [ ] Compare new SO DG forecasts with matched Control samples for each lead.
- [ ] Record actual training command, configuration file, code version, random seed, and result paths.

## 2026-11-06 to 2026-11-11 — Manuscript results added to `analysis/`

- **Supplementary full-dataset table** (`analysis/Supplementary_Table_Full_Dataset_Metrics_KS/`): former S2/S3 merged into one 6/12/18/24 h table, then merged with the pairwise KS / winning-ratio table. Metrics recomputed from the full-year CSVs: 125 of 129 published values unchanged; Simple NN 6 h SS 0.228 → 0.231 (CI 0.229–0.234) and Linear 18 h SS upper CI 0.180 → 0.179. SS uses the Control/Leakage complete-case cohort; MAE/Bias/STD use each model's own samples.
- **Figure S1** (`analysis/Figure_S1_IFS_ERA5_Local_Minimum/`): original storm identified as `JJA_2010_NH_neg_135`, raw indices 4–7 (15 Jul 2010 18 UTC – 16 Jul 12 UTC); four separate panels + shared colorbar.
- **Figure 4d**: x axis clarified as available forecast samples × 6 h (not lifetime); variance diagnostic and variance-standardized r added (slope reduction ~79%, descriptive).
- **Figure 7** (`analysis/Figure_07_Domain_Generalization/`): Control vs DG MAE by held-out season and region, matched on `(storm_key, step_index, lead_h)`.
- **Table 1 / Figure 1**: `table1_fig1.py` added (Control cohort, odd time indices, revised SA/SO lists).

### Open items

- [ ] Recompute pairwise winning ratios and KS distances from forecast-level data; describe p-values as nominal (no within-storm dependence correction).
- [ ] Decide on Figure S1 minimum search (circular 8° vs original ±8-cell square, `--legacy-square-8`).
- [ ] Decide whether to show MSL-only 6 h SS (0.513) in the supplementary table.

## Experiment entry template

```
Experiment ID:
Date:
Git commit:
Lead time (h):
Train / validation / test storm-ID sources:
Configuration:
Data filters:
Random seed:
Expected and effective test storm count:
Expected and effective forecast step count:
Checkpoint:
Metrics and matching cohort:
Figure or table produced:
Notes:
```
