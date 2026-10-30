# Short-Term Extratropical Cyclone Intensity Prediction

Research repository for reproducible short-term (6, 12, 18, and 24 h) prediction of extratropical-cyclone central mean sea-level pressure from cyclone-centered ERA5 fields.

**Status:** Research in progress. This repository is intended to start as private. The manuscript, raw ERA5 arrays, model checkpoints, and large evaluation CSVs are not included here.

## Scope

- Data preprocessing and cyclone track / region classifications
- Convolutional neural network (Control, Leakage, and ablation experiments)
- Persistence and linear regression baselines; 2010 IFS comparison
- Seasonal, regional, and hemisphere domain-generalization (DG) experiments
- Regeneration of manuscript tables and figures, with matched-sample evaluation

## Layout

| Path | Purpose |
| --- | --- |
| `classifications/` | Versioned storm-ID lists and overlap audit |
| `src/` | Original model and data-loading source code (to be added) |
| `configs/` | Redacted, reproducible training configuration files (to be added) |
| `scripts/` | Data integrity and reproducibility utilities |
| `analysis/` | Code generating figures, tables and statistical tests (to be added) |
| `docs/` | Dataset conventions, decisions and reproduction notes |
| `data/` | Local untracked data (never commit ERA5/tensors) |
| `outputs/` | Local untracked runs, plots, checkpoints and logs |

## First validation

Run from the repository root:

```bash
python scripts/audit_classifications.py \
  --left classifications/revised/classification_SH_SA_corrected.txt \
  --right classifications/revised/classification_SH_SO_exclusive.txt \
  --require-disjoint
```

This verifies that the revised South Atlantic and Southern Ocean storm lists have no shared storm IDs. It does **not** establish whether a storm has complete variable files or passes filtering in the training loader.

## Reproducibility checklist

Before treating an experiment as final, record (in `docs/experiments.md`): the exact git commit, configuration, training seed, track-ID split, spatial/temporal filters, evaluation cohort (including lead time), number of storms and forecast steps, and source figure/table script.

The canonical filtering rules must be validated against the *actual* current training pipeline. A `control_34.csv` membership check alone is not a universal filter for 6/12/18/24 h experiments.

## Paper

Working title: *Short-Term Prediction of Extratropical Cyclone Intensity from Cyclone-Centered Atmospheric Fields Using Deep Learning*.

Citations, authorship, final supplementary figure numbering, and public data availability statements will be finalized alongside the manuscript.
