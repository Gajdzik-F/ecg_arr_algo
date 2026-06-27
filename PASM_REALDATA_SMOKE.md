# PASM-Rhythm Real-Data Summary

This report combines the configured WFDB/PhysioNet real-data preset.
It is a research validation checkpoint, not clinical certification.

Preset: `smoke`
Records loaded: 3
Records with truth episodes: 3

## Loaded Record Inventory

| record_id | duration_s | beats | truth_episodes | truth_duration_s | truth_types |
| --- | --- | --- | --- | --- | --- |
| mitdb/200 | 900.000 | 1328 | 2 | 2.244 | ectopic_like |
| afdb/04015 | 900.000 | 1464 | 3 | 203.800 | af_like |
| afdb/04126 | 900.000 | 1893 | 2 | 586.000 | af_like |

## Summary

| model | episode_f1_mean | episode_precision_mean | episode_recall_mean | false_alarms_per_hour_mean | typed_f1_mean |
| --- | --- | --- | --- | --- | --- |
| pasm_physionet | 0.822 | 1.000 | 0.722 | 0.000 | 0.964 |

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

## Per-Record Type Metrics

| record_id | type | precision | recall | f1 | mean_iou |
| --- | --- | --- | --- | --- | --- |
| afdb/04015 | af_like | 1.000 | 0.667 | 0.800 | 0.823 |
| afdb/04015 | ectopic_like | 1.000 | 1.000 | 1.000 |  |
| afdb/04015 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| afdb/04015 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| afdb/04015 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |
| afdb/04126 | af_like | 1.000 | 1.000 | 1.000 | 0.916 |
| afdb/04126 | ectopic_like | 1.000 | 1.000 | 1.000 |  |
| afdb/04126 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| afdb/04126 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| afdb/04126 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |
| mitdb/200 | af_like | 1.000 | 1.000 | 1.000 |  |
| mitdb/200 | ectopic_like | 1.000 | 0.500 | 0.667 | 1.000 |
| mitdb/200 | noise_uncertain | 1.000 | 1.000 | 1.000 |  |
| mitdb/200 | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| mitdb/200 | sinus_tachy | 1.000 | 1.000 | 1.000 |  |

## Per-Record Episode Metrics

| record_id | tp_macro | fp_macro | fn_macro | precision_macro | recall_macro | f1_macro | f1_false_alarms_per_hour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| afdb/04015 | 2 | 0 | 1 | 1.000 | 0.667 | 0.800 | 0.000 |
| afdb/04126 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| mitdb/200 | 1 | 0 | 1 | 1.000 | 0.500 | 0.667 | 0.000 |

## Skipped Or Empty Records

_None._

## Current Interpretation

- The preset exercises both AFDB rhythm annotations and MITDB short ectopic runs.
- AFDB false alarms are much lower after PhysioNet evidence postprocessing; validate these parameters on a broader patient-wise split.
- MITDB ectopy has stricter patient-relative short-RR evidence, but `mitdb/203` remains the main false-alarm stress case.
- This is still a research checkpoint; the next step is patient-wise train/holdout validation.
