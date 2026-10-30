# Input and cyclone naming conventions

Storm IDs are represented as `.npy` file names such as `DJF_1979_NH_neg_52.npy`. Some historical classification exports included a doubled suffix (`.npy.npy`). The audit script normalizes one or more `.npy` suffixes **for comparison only**; the stored original files are preserved for provenance.

The historical tensor layout observed in the project code places an individual storm under `<base>/<parameter>/<year>/<storm.npy>`; pressure-level fields are nested, for example `<base>/ta/850/<year>/<storm.npy>`. Encoded channels (`date#sin`, `date#cos`, `lon#sin`, and `lon#cos`) share the respective base `date` / `lon` files. Confirm the current data loader before running final experiments.

Regional membership and actual data availability are separate concepts. Storms absent from a given Control forecast CSV may have been excluded by genesis-latitude filters, required-input availability, forecast-lead-specific step selection, or other criteria. Do not assume the reason solely from their absence in a CSV.

**Key distinction:** Cyclone count, forecast-step count, and the number of time samples for a specific lead are different measures. Do not infer one from another.
