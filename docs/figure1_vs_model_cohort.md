# Figure 1 versus the model evaluation cohort

Figure 1 in the manuscript is a broad climatological map of storm-track density and was retained for geographic context. The older tracking-based density script does not apply the model's full-track maximum latitude restriction (70 degrees). Consequently, the map includes high-latitude storm paths not used for CNN training/evaluation.

Table 1 instead describes the actual Control cohort: 45,378 unique storm IDs, covering 821,014 six-hour positions in the reconstructed full tracks (NH 21,629; SH 23,749). The six-hour sampling is determined from odd `t` indices in the original three-hour track files. The revised regional classification removes South Atlantic IDs from the Southern Ocean group.

The scripts and cohort manifest are in `analysis/table1_figure1/`.

**Before publication**: verify the date-range provenance of the manuscript figure. The recovered source script skips 1979 although the original figure filename/caption says 1979–2020. This issue should be resolved in the caption/methods or by generating a correctly dated climatology; it must not be hidden by the model-cohort explanation.
