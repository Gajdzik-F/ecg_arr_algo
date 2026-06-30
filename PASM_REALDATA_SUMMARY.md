# PASM-Rhythm Real-Data Summary

This report combines the configured WFDB/PhysioNet real-data preset.
It is a research validation checkpoint, not clinical certification.

Preset: `mini`
Records loaded: 10
Records with truth episodes: 7

## Loaded Record Inventory

| record_id | duration_s | beats | truth_episodes | truth_duration_s | truth_types |
| --- | --- | --- | --- | --- | --- |
| mitdb/200 | 900.000 | 1328 | 2 | 2.244 | ectopic_like |
| mitdb/201 | 900.000 | 1038 | 1 | 0.978 | ectopic_like |
| mitdb/203 | 900.000 | 1542 | 12 | 11.842 | ectopic_like |
| mitdb/205 | 900.000 | 1379 | 2 | 5.364 | ectopic_like |
| mitdb/208 | 900.000 | 1503 | 0 | 0.000 |  |
| afdb/04015 | 1200.000 | 1874 | 3 | 203.800 | af_like |
| afdb/04043 | 1200.000 | 2149 | 1 | 134.008 | af_like |
| afdb/04048 | 1200.000 | 1382 | 0 | 0.000 |  |
| afdb/04126 | 1200.000 | 2492 | 3 | 774.856 | af_like |
| afdb/04746 | 1200.000 | 1278 | 0 | 0.000 |  |

## Summary

| model | episode_f1_mean | episode_precision_mean | episode_recall_mean | false_alarms_per_hour_mean | typed_f1_mean |
| --- | --- | --- | --- | --- | --- |
| pasm_physionet | 0.710 | 0.786 | 0.667 | 5.143 | 0.942 |

## Evidence Layer Parameters

- AF merge gap: 45.0 s
- AF-adjacent tachy suppression margin: 10.0 s
- Minimum retained sinus tachy duration: 3.0 s
- Ectopy short RR: 0.50 s
- Ectopy relative RR fraction: 0.75
- Ectopy merge gap: 1.0 s
- Ectopy flood rate threshold: 30.0 / h
- Ectopy flood minimum confidence: 0.40
- Ectopy flood density window: 10.0 s
- Ectopy flood minimum density: 6
- Ectopy flood minimum candidates: 10
- Ectopy flood strong morph z: 0.55
- Ectopy flood dense morph z: 0.60

## Per-Record Type Metrics

| record_id | type | precision | recall | f1 | mean_iou |
| --- | --- | --- | --- | --- | --- |
| afdb/04015 | af_like | 1.000 | 0.667 | 0.800 | 0.823 |
| afdb/04015 | ectopic_like | 1.000 | 1.000 | 1.000 |  |
| afdb/04015 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| afdb/04015 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| afdb/04015 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |
| afdb/04043 | af_like | 1.000 | 1.000 | 1.000 | 0.941 |
| afdb/04043 | ectopic_like | 1.000 | 1.000 | 1.000 |  |
| afdb/04043 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| afdb/04043 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| afdb/04043 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |
| afdb/04126 | af_like | 1.000 | 1.000 | 1.000 | 0.924 |
| afdb/04126 | ectopic_like | 1.000 | 1.000 | 1.000 |  |
| afdb/04126 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| afdb/04126 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| afdb/04126 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |
| mitdb/200 | af_like | 1.000 | 1.000 | 1.000 |  |
| mitdb/200 | ectopic_like | 1.000 | 0.500 | 0.667 | 1.000 |
| mitdb/200 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| mitdb/200 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| mitdb/200 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |
| mitdb/201 | af_like | 1.000 | 1.000 | 1.000 |  |
| mitdb/201 | ectopic_like | 0.000 | 0.000 | 0.000 |  |
| mitdb/201 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| mitdb/201 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| mitdb/201 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |
| mitdb/203 | af_like | 1.000 | 1.000 | 1.000 |  |
| mitdb/203 | ectopic_like | 0.500 | 0.500 | 0.500 | 0.818 |
| mitdb/203 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| mitdb/203 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| mitdb/203 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |
| mitdb/205 | af_like | 1.000 | 1.000 | 1.000 |  |
| mitdb/205 | ectopic_like | 1.000 | 1.000 | 1.000 | 0.800 |
| mitdb/205 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| mitdb/205 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| mitdb/205 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |

## Per-Record Episode Metrics

| record_id | tp_macro | fp_macro | fn_macro | precision_macro | recall_macro | f1_macro | f1_false_alarms_per_hour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| afdb/04015 | 2 | 0 | 1 | 1.000 | 0.667 | 0.800 | 0.000 |
| afdb/04043 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| afdb/04126 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| mitdb/200 | 1 | 0 | 1 | 1.000 | 0.500 | 0.667 | 0.000 |
| mitdb/201 | 0 | 3 | 1 | 0.000 | 0.000 | 0.000 | 12.000 |
| mitdb/203 | 6 | 6 | 6 | 0.500 | 0.500 | 0.500 | 24.000 |
| mitdb/205 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |

## Candidate-Level Metrics

These rows evaluate the physiology-guided candidate generator before final deterministic filtering or AI/reranker acceptance.

| record_id | candidate_count | candidate_tp_rows | candidate_fp_rows | truth_episodes | candidate_recalled_truth | truth_never_proposed | candidate_precision | candidate_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mitdb/200 | 375 | 9 | 366 | 2 | 2 | 0 | 0.024 | 1.000 |
| mitdb/201 | 251 | 2 | 249 | 1 | 1 | 0 | 0.008 | 1.000 |
| mitdb/203 | 1117 | 30 | 1087 | 12 | 12 | 0 | 0.027 | 1.000 |
| mitdb/205 | 78 | 19 | 59 | 2 | 2 | 0 | 0.244 | 1.000 |
| afdb/04015 | 16 | 2 | 14 | 3 | 2 | 1 | 0.125 | 0.667 |
| afdb/04043 | 29 | 1 | 28 | 1 | 1 | 0 | 0.034 | 1.000 |
| afdb/04126 | 113 | 3 | 110 | 3 | 3 | 0 | 0.027 | 1.000 |

## False-Negative Stage Audit

Each FN is assigned to the earliest visible stage that can explain the miss.

| record_id | episode_time | start_s | end_s | label_type | generated_candidate_yes_no | rejected_stage | best_candidate_iou | best_final_iou | source | rr_pattern | rr_support | morphology_score | sqi | density | pause_support | ectopy_pattern | final_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mitdb/200 | 109.219 | 109.219 | 110.531 | ectopic_like | yes | deterministic_rules_rejected | 0.650 | 0.000 | relaxed_ectopy | premature_plus_pause | 0.083 | 1.808 | 1.000 | 1.000 | 0.744 | premature_plus_pause | FN |
| mitdb/201 | 477.453 | 477.453 | 478.431 | ectopic_like | yes | deterministic_rules_rejected | 0.469 | 0.000 | relaxed_ectopy | morphology_cluster | 0.056 | 1.314 | 1.000 | 1.000 | 0.000 | morphology_cluster | FN |
| mitdb/203 | 46.461 | 46.461 | 47.156 | ectopic_like | yes | deterministic_rules_rejected | 1.000 | 0.000 | relaxed_ectopy | short_coupled_run | 0.240 | 0.604 | 1.000 | 1.000 | 0.800 | short_coupled_run | FN |
| mitdb/203 | 92.033 | 92.033 | 92.792 | ectopic_like | yes | deterministic_rules_rejected | 1.000 | 0.000 | relaxed_ectopy | short_coupled_run | 0.124 | 0.633 | 1.000 | 1.000 | 0.947 | short_coupled_run | FN |
| mitdb/203 | 269.125 | 269.125 | 269.864 | ectopic_like | yes | deterministic_rules_rejected | 1.000 | 0.000 | relaxed_ectopy | short_coupled_run | 0.150 | 0.625 | 1.000 | 1.000 | 0.596 | short_coupled_run | FN |
| mitdb/203 | 303.825 | 303.825 | 306.819 | ectopic_like | yes | deterministic_rules_rejected | 1.000 | 0.000 | relaxed_ectopy | rr_irregular_burst | 0.317 | 0.361 | 1.000 | 1.000 | 1.000 | rr_irregular_burst | FN |
| mitdb/203 | 511.097 | 511.097 | 511.803 | ectopic_like | yes | deterministic_rules_rejected | 0.318 | 0.000 | relaxed_ectopy | short_coupled_run | 0.188 | 0.538 | 1.000 | 1.000 | 1.000 | short_coupled_run | FN |
| mitdb/203 | 758.111 | 758.111 | 759.103 | ectopic_like | yes | deterministic_rules_rejected | 0.664 | 0.000 | relaxed_ectopy | short_coupled_run | 0.279 | 0.714 | 1.000 | 1.000 | 0.000 | short_coupled_run | FN |
| afdb/04015 | 487.092 | 487.092 | 488.776 | af_like | no | label_matching_or_episode_boundary | 0.023 | 0.023 | baseline | baseline | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | baseline | FN |

## Skipped Or Empty Records

| record_id | reason |
| --- | --- |
| mitdb/208 | empty_truth |
| afdb/04048 | empty_truth |
| afdb/04746 | empty_truth |

## Diagnostic Sidecars

When generated through the CLI, per-record TP/FP/FN diagnostics are written under `reports/diagnostics/`.

## Current Interpretation

- The preset exercises both AFDB rhythm annotations and MITDB short ectopic runs.
- AFDB false alarms are much lower after PhysioNet evidence postprocessing; validate these parameters on a broader patient-wise split.
- The pipeline is reported as four separate stages: physiology-guided candidate generator, deterministic PASM evidence/fallback, PASM-AI reranker/acceptor, and final decision fusion.
- MITDB candidate coverage should be interpreted before final F1; the reranker cannot recover truth episodes that were never proposed.
- MITDB ectopy has stricter patient-relative short-RR evidence, but `mitdb/203` remains the main false-alarm stress case.
- This is still a research checkpoint; the next step is patient-wise train/holdout validation.
