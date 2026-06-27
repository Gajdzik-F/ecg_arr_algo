# PASM-Rhythm Synthetic Validation

This report is generated from deterministic synthetic ECG rhythm cohorts.
It validates the PASM-Rhythm code path and episode metrics, but it is not clinical validation.

Records: 30

## Decoder Thresholds

| state | min_confidence |
| --- | --- |
| af_like | 0.400 |
| ectopic_like | 0.280 |
| noise_uncertain | 0.300 |
| sinus_brady | 0.240 |
| sinus_tachy | 0.240 |

## Training Summary

| model | episode_f1_mean | episode_precision_mean | episode_recall_mean | false_alarms_per_hour_mean | typed_f1_mean |
| --- | --- | --- | --- | --- | --- |
| pasm_tuned | 0.847 | 0.804 | 0.917 | 15.383 | 0.889 |

## Summary

| model | episode_f1_mean | episode_precision_mean | episode_recall_mean | false_alarms_per_hour_mean | typed_f1_mean |
| --- | --- | --- | --- | --- | --- |
| pasm_tuned | 0.834 | 0.773 | 0.917 | 17.581 | 0.883 |
| pasm_default | 0.749 | 0.733 | 0.792 | 19.256 | 0.804 |

## Per-Type Mean Metrics

| model | type | precision | recall | f1 | mean_iou |
| --- | --- | --- | --- | --- | --- |
| pasm_default | af_like | 0.419 | 0.600 | 0.474 | 0.585 |
| pasm_default | ectopic_like | 1.000 | 1.000 | 1.000 | 0.997 |
| pasm_default | noise_uncertain | 1.000 | 1.000 | 1.000 | 1.000 |
| pasm_default | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| pasm_default | sinus_tachy | 0.933 | 0.567 | 0.544 | 0.669 |
| pasm_tuned | af_like | 0.569 | 0.900 | 0.669 | 0.617 |
| pasm_tuned | ectopic_like | 1.000 | 1.000 | 1.000 | 0.997 |
| pasm_tuned | noise_uncertain | 1.000 | 1.000 | 1.000 | 1.000 |
| pasm_tuned | sinus_brady | 1.000 | 1.000 | 1.000 |  |
| pasm_tuned | sinus_tachy | 0.933 | 0.767 | 0.744 | 0.695 |

## Interpretation

- This report validates the PASM-only rhythm path on synthetic ectopic, AF-like, tachy, brady, and noise episodes.
- Results are useful as a regression gate before moving to PhysioNet/MIT-BIH style validation.
- The next validation stage must use patient-wise splits on real annotated ECG databases.
