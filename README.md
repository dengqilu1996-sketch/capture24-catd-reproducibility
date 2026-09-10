# Capture-24 CATD reproducibility code

This repository contains only the code and run instructions used for the
Capture-24 CATD experiments.

## What is included

- `experiments/prepare_capture24_official_repro.py`: official 10-second window
  preprocessing wrapper.
- `experiments/run_official_baselines.py`: RF/RF+HMM/XGBoost baseline runner.
- `experiments/tune_official_rf_hmm_gamma_locked_test.py`: gamma-tuned official
  hard-observation RF+HMM fairness audit.
- `experiments/tune_rf_temporal_decoding_locked_test.py`: probability-level RF
  temporal decoding and CATD selection/locked-test evaluation.
- `scripts/`: bootstrap uncertainty, conflict diagnostics, null control, and
  locked-test subgroup checks.
- `configs/paths.capture24.example.yaml`: path configuration template.
- `configs/revision_reproducibility.yaml`: consolidated revision settings
  (a reference record, not a replacement for each script's runtime arguments).
- `experiments/run_auxiliary_xgb_labels.py` and
  `experiments/tune_bcm_derivation_locked_test.py`: auxiliary XGBoost and
  internal-validation prerequisite generation.
- `revision_audits/`: CRF, CNN/RNN, PAMAP2, retraining, and runtime audit
  source scripts; read its README before execution.
- `requirements.txt`: minimal Python runtime dependencies.
- `requirements-official-capture24.txt` and
  `environment-official-capture24.yml`: environment references for the
  official Capture-24 implementation.

## Third-party code and data

The official Capture-24 implementation is included as a Git submodule at
`capture24_project/official_code/capture24` and is pinned to commit
`f861b44f5675cb3e8294cd3d560d7a71a749616f` from
https://github.com/OxWearables/capture24.

Download the raw Capture-24 release from the Oxford data record
(https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001).
Raw data are not redistributed here.

## Reproduction outline

Run commands from the repository root in the pinned Python/conda environment:

```powershell
git clone --recurse-submodules <repository-url>
Copy-Item configs/paths.capture24.example.yaml configs/paths.capture24.yaml
# Edit configs/paths.capture24.yaml for the local raw-data path.

python experiments/prepare_capture24_official_repro.py `
  --raw-parent <parent-containing-capture24-folder> `
  --outdir data/prepared_data_official_repro `
  --annots Walmsley2020,WillettsSpecific2018 `
  --winsec 10

python experiments/run_official_baselines.py `
  --config configs/paths.capture24.yaml `
  --models rf,rf_hmm `
  --save-proba

python experiments/tune_official_rf_hmm_gamma_locked_test.py

python experiments/tune_rf_temporal_decoding_locked_test.py `
  --config configs/paths.capture24.yaml `
  --selection-mode conflict_priority `
  --macro-retention 0.95 `
  --smoothing 0.001
```

The tuning protocol is participant-disjoint: P001–P080 estimate mappings and
transitions, P081–P100 select the operating point, and P101–P151 are evaluated
once as the locked test set. The scripts write predictions and metrics to local
`results/` and `paper_artifacts/tables/` directories; those generated outputs
are intentionally excluded from this repository.

## Reproducibility notes

- The principal Capture-24 pathway uses RF fine and coarse heads and an
  XGBoost MET-intensity head, all consuming the same 32-feature input.
  The decoder loads validation intensity probabilities from the BCM tuning
  artifacts and locked-test intensity probabilities from the auxiliary XGBoost
  artifacts. These prerequisites must be generated before running the decoder;
  the outline above is not a complete raw-data-to-results pipeline.
- Mapping and transition matrices use additive smoothing 0.001 and row
  normalization.
- Probabilities are floored at 1e-12 before logarithms; modulated emissions
  are renormalized.
- The paired participant-level bootstrap uses 5,000 resamples and seed
  20260630.
- The conflict metric is an internal label-system diagnostic, not independent
  behavioral ground truth.

## Scope

The principal pathway is RF/RF+HMM/CATD and its diagnostics. Revision audit
scripts are also provided, but retain historical input/path prerequisites and
are not a self-contained end-to-end replication environment. No new training
or independent raw-data-to-results replication was performed for this release.

Revision audit scripts and aggregate tables are supplied separately in
`Supplementary_Reproducibility.zip` with the revised submission. That archive
includes CRF, CNN/RNN, PAMAP2, retraining, and runtime audits and explicitly
documents path and data prerequisites. It excludes raw recordings, participant-
aligned predictions and checkpoints. The public `revision_audits/` directory
contains source code only; aggregate tables and reports remain in the submission
supplement. Manuscript files and figures are not part of this code release.

## Code-only publication policy

The current branch excludes experimental data, observed result values, figures,
prediction arrays and model weights. The ablation-table utility requires a local
`--xgb-rows` CSV instead of embedding results; see `scripts/ABLATION_INPUT.md`.
Only source syntax and input validation were checked for this release, not a
fresh end-to-end experiment. The public branch starts from a clean code-only
snapshot; earlier project history is not part of this branch.
