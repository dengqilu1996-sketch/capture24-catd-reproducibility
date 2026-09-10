# Local input for the ablation-table utility

Run `python scripts/make_aux_temporal_v2_ablation_table.py --xgb-rows <local.csv>`
after generating the required RF prediction artifacts and locally computing the
XGBoost rows. The CSV is deliberately not supplied in this repository.

Required columns: `family`, `method`, `selection_role`, `alpha`, `beta`,
`gamma_or_gate`, `macro_f1`, `balanced_accuracy`, `intensity_conflict`,
`any_conflict`, `corrected`, `harmed`.

Provide exactly two rows with family `XGBoost`: one method `XGB baseline` and
one method `post-hoc BCM`. Metrics must be finite within [0, 1]; counts must be
non-negative integers. Compute corrected/harmed relative to your local XGBoost
baseline on the same aligned evaluation windows. Do not fill in values from
the paper, substitute synthetic values, or combine different data splits.

This utility formats supplied XGBoost results; it does not independently
recompute or verify them. RF results are computed from the referenced local
prediction artifacts. Historical paths and parameter labels must be adapted
and checked against the actual local run before interpreting the table.

The current public source includes no bundled XGBoost observations or fixed
experimental conclusions. Keep input CSVs and generated outputs local.
