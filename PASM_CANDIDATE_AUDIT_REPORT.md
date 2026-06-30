# PASM Candidate Audit Report

Generated on 2026-06-28 for the PASM-AI arrhythmia episode detection project.

## Evaluation Protocol Frozen

- Real-data presets: `mitdb-mini`, `afdb-mini`, and combined `mini`.
- ML/reranker preset: `tiny`, with patient-wise train records `mitdb/200`, `mitdb/205`, `afdb/04015`, `afdb/04043` and holdout records `mitdb/201`, `mitdb/203`, `afdb/04126`.
- Episode matching remains unchanged: typed episode IoU threshold `0.30`.
- MITDB truth remains beat-symbol ectopic-like episodes; MITDB final predictions remain annotation-scoped to `ectopic_like`.
- Baselines before code changes were saved as `PASM_MITDB_MINI_VALIDATION.baseline.md` and `PASM_AFDB_MINI_VALIDATION.baseline.md`.

## Baseline vs Current

| dataset | checkpoint | final_f1 | final_precision | final_recall | fp_per_hour |
| --- | --- | --- | --- | --- | --- |
| MITDB mini | baseline | 0.542 | 0.625 | 0.500 | 9.000 |
| MITDB mini | current | 0.542 | 0.625 | 0.500 | 9.000 |
| AFDB mini | baseline | 0.933 | 1.000 | 0.889 | 0.000 |
| AFDB mini | current | 0.933 | 1.000 | 0.889 | 0.000 |

Final deterministic PASM performance did not change. The implementation intentionally keeps the new permissive candidates out of final acceptance until the reranker/policy can filter them safely.

## Candidate-Level MITDB Metrics

| record_id | candidates | candidate_tp_rows | truth_episodes | truth_never_proposed | candidate_recall |
| --- | --- | --- | --- | --- | --- |
| mitdb/200 | 375 | 9 | 2 | 0 | 1.000 |
| mitdb/201 | 251 | 2 | 1 | 0 | 1.000 |
| mitdb/203 | 1117 | 30 | 12 | 0 | 1.000 |
| mitdb/205 | 78 | 19 | 2 | 0 | 1.000 |

The previous uncovered MITDB holdout episode at `mitdb/203` 303.825-306.819 s is now proposed as an `rr_irregular_burst` candidate with IoU 1.000. This moves the observed MITDB bottleneck from candidate generation coverage to deterministic/reranker acceptance.

## FN Stage Audit

All current MITDB mini false negatives now have generated candidates. Their rejected stage is `deterministic_rules_rejected`, not `candidate_generator_miss`.

| record_id | FN count | dominant rejected_stage | notes |
| --- | --- | --- | --- |
| mitdb/200 | 1 | deterministic_rules_rejected | best candidate is `premature_plus_pause` |
| mitdb/201 | 1 | deterministic_rules_rejected | best candidate is `morphology_cluster` |
| mitdb/203 | 6 | deterministic_rules_rejected | includes `short_coupled_run` and new `rr_irregular_burst` |
| mitdb/205 | 0 | none | no current FN |

Detailed CSV/JSON sidecars:

- `reports/diagnostics/mitdb-mini_candidate_metrics.csv`
- `reports/diagnostics/mitdb-mini_fn_audit.csv`
- `reports/diagnostics/mitdb-mini_fn_audit.json`
- `reports/diagnostics/mini_candidate_metrics.csv`
- `reports/diagnostics/mini_fn_audit.csv`

## Ablation Readout

| ablation | MITDB final F1 | AFDB final F1 | MITDB candidate recall | interpretation |
| --- | --- | --- | --- | --- |
| A deterministic PASM only | 0.542 mini / 0.250 tiny holdout | 0.933 mini / 1.000 tiny holdout | 1.000 mini | Strong AFDB, MITDB misses are acceptance-stage |
| B PASM + permissive candidates | 0.542 final unchanged | 0.933 final unchanged | 1.000 mini | Candidate coverage fixed without changing final decisions |
| C PASM + AI reranker safe | 0.250 tiny holdout | 1.000 tiny holdout | 1.000 holdout | Safe reranker preserves baseline but does not rescue MITDB FNs |
| D PASM + AI reranker + hard negatives | 0.250 tiny holdout | 1.000 tiny holdout | 1.000 holdout | Hard-negative v2 still rejects relaxed MITDB rescues |

## Threshold Changelog

No final-decision thresholds were changed.

Candidate proposal change:

| parameter | old value | new value | reason | observed effect |
| --- | --- | --- | --- | --- |
| `rr_irregular_burst` candidate family | absent | enabled in candidate generator | Bridge short abnormal RR/morphology fragments before reranking | MITDB mini truth-never-proposed reduced to 0 |
| `burst_min_cv` | n/a | 0.28 | Require local RR irregularity | Keeps rule physiology-guided |
| `burst_min_rmssd` | n/a | 0.18 | Require beat-to-beat RR variability | Keeps rule physiology-guided |
| `burst_min_abnormal` | n/a | 4 beats | Require repeated short/pause support | Avoids isolated one-beat noise proposals |
| `burst_min_morph_z` | n/a | 0.35 | Allow borderline morphology while exposing score to reranker | Covers fragmented MITDB 203 FN |

## Current Conclusion

The bottleneck is no longer candidate generation coverage on MITDB mini. It is candidate acceptance: permissive candidates include true MITDB episodes, but naive final acceptance creates too many false positives. The next accepted change should tune the explainable reranker/pattern policy against record-wise validation so relaxed candidates can be rescued selectively.

This checkpoint does not satisfy the final success criterion of meaningful MITDB final F1 improvement yet. It does satisfy the intermediate requirement to expose candidate recall, classify FNs by pipeline stage, preserve AFDB performance, and keep deterministic PASM as the core fallback.
