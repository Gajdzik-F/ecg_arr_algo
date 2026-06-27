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

## Skipped Or Empty Records

| record_id | reason |
| --- | --- |
| mitdb/208 | empty_truth |
| afdb/04048 | empty_truth |
| afdb/04746 | empty_truth |

## Current Interpretation

- The preset exercises both AFDB rhythm annotations and MITDB short ectopic runs.
- AFDB false alarms are much lower after PhysioNet evidence postprocessing; validate these parameters on a broader patient-wise split.
- MITDB ectopy has stricter patient-relative short-RR evidence, but `mitdb/203` remains the main false-alarm stress case.
- This is still a research checkpoint; the next step is patient-wise train/holdout validation.
