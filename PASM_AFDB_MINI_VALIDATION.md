# PASM-Rhythm Real-Data Summary

This report combines the configured WFDB/PhysioNet real-data preset.
It is a research validation checkpoint, not clinical certification.

Preset: `afdb-mini`
Records loaded: 5
Records with truth episodes: 3

## Loaded Record Inventory

| record_id | duration_s | beats | truth_episodes | truth_duration_s | truth_types |
| --- | --- | --- | --- | --- | --- |
| afdb/04015 | 1200.000 | 1874 | 3 | 203.800 | af_like |
| afdb/04043 | 1200.000 | 2149 | 1 | 134.008 | af_like |
| afdb/04048 | 1200.000 | 1382 | 0 | 0.000 |  |
| afdb/04126 | 1200.000 | 2492 | 3 | 774.856 | af_like |
| afdb/04746 | 1200.000 | 1278 | 0 | 0.000 |  |

## Summary

| model | episode_f1_mean | episode_precision_mean | episode_recall_mean | false_alarms_per_hour_mean | typed_f1_mean |
| --- | --- | --- | --- | --- | --- |
| pasm_physionet | 0.933 | 1.000 | 0.889 | 0.000 | 0.987 |

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

## Per-Record Episode Metrics

| record_id | tp_macro | fp_macro | fn_macro | precision_macro | recall_macro | f1_macro | f1_false_alarms_per_hour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| afdb/04015 | 2 | 0 | 1 | 1.000 | 0.667 | 0.800 | 0.000 |
| afdb/04043 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| afdb/04126 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |

## Candidate-Level Metrics

These rows evaluate the physiology-guided candidate generator before final deterministic filtering or AI/reranker acceptance.

| record_id | candidate_count | candidate_tp_rows | candidate_fp_rows | truth_episodes | candidate_recalled_truth | truth_never_proposed | candidate_precision | candidate_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afdb/04015 | 16 | 2 | 14 | 3 | 2 | 1 | 0.125 | 0.667 |
| afdb/04043 | 29 | 1 | 28 | 1 | 1 | 0 | 0.034 | 1.000 |
| afdb/04126 | 113 | 3 | 110 | 3 | 3 | 0 | 0.027 | 1.000 |

## False-Negative Stage Audit

Each FN is assigned to the earliest visible stage that can explain the miss.

| record_id | episode_time | start_s | end_s | label_type | generated_candidate_yes_no | rejected_stage | best_candidate_iou | best_final_iou | source | rr_pattern | rr_support | morphology_score | sqi | density | pause_support | ectopy_pattern | final_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afdb/04015 | 487.092 | 487.092 | 488.776 | af_like | no | label_matching_or_episode_boundary | 0.023 | 0.023 | baseline | baseline | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | baseline | FN |

## Skipped Or Empty Records

| record_id | reason |
| --- | --- |
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
