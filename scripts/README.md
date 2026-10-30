# Validation scripts

The audit tool uses only the Python standard library.

```bash
# Check revised SA / SO disjointness
python scripts/audit_classifications.py \
  --left classifications/revised/classification_SH_SA_corrected.txt \
  --right classifications/revised/classification_SH_SO_exclusive.txt \
  --require-disjoint

# Compare a classification to a locally available Control CSV (for diagnostics only)
python scripts/audit_classifications.py \
  --left classifications/original/classification_NH_NA.txt \
  --control data/control_34.csv \
  --missing-out outputs/NA_not_in_control34.txt
```

The optional CSV check reports membership in **that CSV only**; it is not a replacement for validating the model's actual eligibility and split logic.
