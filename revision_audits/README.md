# Revision audit source code

These are the source scripts archived for the September 2026 revision.
No raw data, participant-aligned predictions, model weights, manuscript files,
figures or aggregate result tables are included in this directory.

## Scope

- `deep_context/`: current 10-class multi-head CNN and bidirectional-GRU audits.
- `structured_baseline/`: linear-chain CRF feasibility audit with an explicitly
  limited 200-window-per-participant sample, not a full-data comparator.
- `external_validation/`: participant-disjoint PAMAP2 protocol-domain audit.
- `retraining_audit/`: participant-level RF retraining sensitivity.
- `runtime_audit/`: decoder timing and memory measurement.

The main Capture-24 model uses RF fine/coarse heads and XGBoost intensity.
PAMAP2 and RF-retraining audits instead use three RF heads. They are different
experiment scopes and should not be substituted for the principal result.

## Before running

1. Obtain the datasets from their original providers under applicable terms.
2. Initialize the official Capture-24 submodule and install the root environment
   dependencies. Deep audits also require PyTorch; CRF requires sklearn-crfsuite.
3. Inspect each script's imports, input declarations and output paths. These
   historical scripts retain original project roots and timestamped input names;
   adapt them to your local prepared-data layout before execution.
4. Prepare the referenced label-field arrays, posteriors or checkpoints first.
   They are not redistributed. The CNN completion script requires the preceding
   training checkpoint. Runtime audits require saved probabilities. Some scripts
   import helpers from the principal experiment scripts, so adjust import paths
   or place the audit scripts in that experiment directory when reproducing.
5. Consult `../configs/revision_reproducibility.yaml` for recorded settings and
   the submission supplement for aggregate results and execution reports.

Only Python syntax has been checked for this source release. This is not a
claim of fresh end-to-end replication or universal portability. Avoid running
scripts before checking output paths, since historical scripts write results.
