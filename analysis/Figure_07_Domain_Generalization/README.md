# Figure 7 — Domain generalization: MAE vs lead time

MAE of the Control model (solid) and the domain-generalization (DG) models (dashed) at 6, 12, 18 and 24 h. (a) Held-out seasons; (b) held-out regions (NA, NP, MED, SA, SO).

## Files

- `generate_figure_07.py` — writes `MAE_vs_Lead_Seasons.pdf` (panel a) and `MAE_vs_Lead_Regions.pdf` (panel b).
- `Figure_07_MAE_vs_Lead_Control_vs_DG.pdf` — the manuscript figure (both panels).

## Run

From the repository root:

```bash
python analysis/Figure_07_Domain_Generalization/generate_figure_07.py \
  --base-dir data/updated_data --out-dir outputs/figure_07
```

Expected inputs (not committed):

```
data/updated_data/control/control_31.csv ... control_34.csv
data/updated_data/dg/<6|12|18|24>/<sa|so|np|na|med|winter|spring|summer|autumn>.csv
```

Each CSV needs `storm_key`, `step_index`, `dl_abs_error`.

## Method

- For every DG category and lead, DG errors are inner-joined to Control on `(storm_key, step_index, lead_h)`. The Control curve for a category therefore uses exactly the same storms/steps as that DG model (the held-out test set), not the full Control sample.
- MAE = mean of `dl_abs_error` over the matched steps. No confidence intervals are drawn.

## Notes

- `storm_key` is only whitespace-stripped before the join. If Control and DG files differ in suffix (e.g. `.npy`), steps will silently drop out — check matched counts.
- SA and SO results depend on the revised, disjoint storm lists (`classifications/revised/`); the DG test storms must not appear in the training regions.
