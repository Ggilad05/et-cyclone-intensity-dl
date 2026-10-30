# Regional cyclone lists

`original/` contains the historical region-membership files as supplied, without correction. `revised/` contains the current revised South Atlantic and Southern Ocean lists.

The original Southern Ocean membership includes storms that also belong to South Atlantic. The revised Southern Ocean file removes those shared storm IDs; the revised South Atlantic list retains the full South Atlantic set (with a corrected `.npy.npy` suffix).

Do not use `classification_SH_SA_exclusive.txt` (354 storms) as the SA test cohort: that was an exploratory alternative which the project did not adopt.

The NA/NP/MED lists have not been silently rewritten using a single lead-time Control CSV; determine the canonical eligible storm cohort from the real data pipeline before standardizing all lists for manuscript figures/tables.
