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
