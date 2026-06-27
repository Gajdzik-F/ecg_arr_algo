# PASM-Rhythm ML Validation

This report evaluates lightweight NumPy softmax decoders on PASM feature tables.
It is a research checkpoint, not clinical validation.

Preset: `mini`

## Patient-Wise Split

Requested train records:

| record_id |
| --- |
| mitdb/200 |
| mitdb/205 |
| afdb/04015 |
| afdb/04043 |
| afdb/04048 |

Requested holdout records:

| record_id |
| --- |
| mitdb/201 |
| mitdb/203 |
| mitdb/208 |
| afdb/04126 |
| afdb/04746 |

Informative train records:

| record_id |
| --- |
| mitdb/200 |
| mitdb/205 |
| afdb/04015 |
| afdb/04043 |

Informative holdout records:

| record_id |
| --- |
| mitdb/201 |
| mitdb/203 |
| afdb/04126 |

Skipped records:

| split | record_id | reason |
| --- | --- | --- |
| train | afdb/04048 | empty_truth |
| holdout | mitdb/208 | empty_truth |
| holdout | afdb/04746 | empty_truth |

## Beat Label Counts

Train:

| label | beats |
| --- | --- |
| af_like | 710 |
| ectopic_like | 25 |
| normal | 5993 |

Holdout:

| label | beats |
| --- | --- |
| af_like | 1877 |
| ectopic_like | 49 |
| normal | 3142 |

## Tuned Raw ML Decoder Thresholds

| state | min_confidence |
| --- | --- |
| af_like | 0.500 |
| ectopic_like | 0.280 |
| noise_uncertain | 0.300 |
| sinus_brady | 0.240 |
| sinus_tachy | 0.240 |

## Tuned FP-Aware Decoder Thresholds

| state | min_confidence |
| --- | --- |
| af_like | 0.400 |
| ectopic_like | 0.280 |
| noise_uncertain | 0.300 |
| sinus_brady | 0.240 |
| sinus_tachy | 0.240 |

## Hard Negative Training

| hard_negative_beats | hard_negative_boost |
| --- | --- |
| 151.000 | 3.000 |

## Tuned Guard Config

| parameter | value |
| --- | --- |
| normal_bias | 0.180 |
| min_episode_confidence | 0.650 |
| min_episode_sqi | 0.500 |
| ectopy_min_morph_z | 0.550 |
| ectopy_min_delta_rr_z_abs | 3.000 |
| ectopy_min_score | 0.450 |
| min_beats_af_like | 12.000 |
| min_beats_ectopic_like | 4.000 |
| min_beats_noise_uncertain | 3.000 |
| min_beats_sinus_brady | 8.000 |
| min_beats_sinus_tachy | 8.000 |

## Train Summary

| model | episode_f1_mean | episode_precision_mean | episode_recall_mean | false_alarms_per_hour_mean | typed_f1_mean |
| --- | --- | --- | --- | --- | --- |
| pasm_physionet | 0.867 | 1.000 | 0.792 | 0.000 | 0.973 |
| pasm_ml_decoder_fpaware | 0.508 | 0.688 | 0.667 | 5.250 | 0.902 |
| pasm_ml_decoder_guarded | 0.483 | 0.667 | 0.667 | 8.250 | 0.848 |
| pasm_ml_decoder | 0.472 | 0.658 | 0.667 | 10.500 | 0.845 |

## Holdout Summary

| model | episode_f1_mean | episode_precision_mean | episode_recall_mean | false_alarms_per_hour_mean | typed_f1_mean |
| --- | --- | --- | --- | --- | --- |
| pasm_physionet | 0.500 | 0.500 | 0.500 | 12.000 | 0.900 |
| pasm_ml_decoder | 0.222 | 0.500 | 0.333 | 4.333 | 0.844 |
| pasm_ml_decoder_fpaware | 0.200 | 0.810 | 0.333 | 4.000 | 0.840 |
| pasm_ml_decoder_guarded | 0.200 | 0.810 | 0.333 | 4.000 | 0.840 |

## Holdout Per-Record Metrics

| record_id | model | tp | fp | fn | precision | recall | f1 | false_alarms_per_hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afdb/04126 | pasm_ml_decoder | 3 | 3 | 0 | 0.500 | 1.000 | 0.667 | 9.000 |
| afdb/04126 | pasm_ml_decoder_fpaware | 3 | 4 | 0 | 0.429 | 1.000 | 0.600 | 12.000 |
| afdb/04126 | pasm_ml_decoder_guarded | 3 | 4 | 0 | 0.429 | 1.000 | 0.600 | 12.000 |
| afdb/04126 | pasm_physionet | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| mitdb/201 | pasm_ml_decoder | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | 4.000 |
| mitdb/201 | pasm_ml_decoder_fpaware | 0 | 0 | 1 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/201 | pasm_ml_decoder_guarded | 0 | 0 | 1 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/201 | pasm_physionet | 0 | 3 | 1 | 0.000 | 0.000 | 0.000 | 12.000 |
| mitdb/203 | pasm_ml_decoder | 0 | 0 | 12 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/203 | pasm_ml_decoder_fpaware | 0 | 0 | 12 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/203 | pasm_ml_decoder_guarded | 0 | 0 | 12 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/203 | pasm_physionet | 6 | 6 | 6 | 0.500 | 0.500 | 0.500 | 24.000 |

## Holdout FP/h By Record

| record_id | model | false_alarms_per_hour |
| --- | --- | --- |
| afdb/04126 | pasm_ml_decoder | 9.000 |
| afdb/04126 | pasm_ml_decoder_fpaware | 12.000 |
| afdb/04126 | pasm_ml_decoder_guarded | 12.000 |
| afdb/04126 | pasm_physionet | 0.000 |
| mitdb/201 | pasm_ml_decoder | 4.000 |
| mitdb/201 | pasm_ml_decoder_fpaware | 0.000 |
| mitdb/201 | pasm_ml_decoder_guarded | 0.000 |
| mitdb/201 | pasm_physionet | 12.000 |
| mitdb/203 | pasm_ml_decoder | 0.000 |
| mitdb/203 | pasm_ml_decoder_fpaware | 0.000 |
| mitdb/203 | pasm_ml_decoder_guarded | 0.000 |
| mitdb/203 | pasm_physionet | 24.000 |

## Holdout False Positives By Type

| model | type | false_positives |
| --- | --- | --- |
| pasm_ml_decoder | af_like | 3 |
| pasm_ml_decoder | ectopic_like | 1 |
| pasm_ml_decoder_fpaware | af_like | 3 |
| pasm_ml_decoder_guarded | af_like | 4 |
| pasm_physionet | ectopic_like | 9 |

## Holdout False Positives By Failure Stage

| model | failure_stage | false_positives |
| --- | --- | --- |
| pasm_ml_decoder | beat_state_scoring | 4 |
| pasm_ml_decoder_fpaware | beat_state_scoring | 3 |
| pasm_ml_decoder_guarded | beat_state_scoring | 4 |
| pasm_physionet | decoder_low_confidence | 8 |
| pasm_physionet | decoder_short_episode | 1 |

## FP Removed By Guard Reason

| model | type | reason | removed |
| --- | --- | --- | --- |
| pasm_ml_decoder_fpaware | af_like | too_few_beats | 14 |
| pasm_ml_decoder_guarded | af_like | too_few_beats | 14 |

## Top Holdout False-Positive Episodes

| record_id | model | type | start_s | end_s | duration_s | confidence | mean_sqi | beats | best_iou | failure_stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afdb/04126 | pasm_ml_decoder | af_like | 531.912 | 555.476 | 23.564 | 0.977 | 1.000 | 57 | 0.041 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder | af_like | 49.208 | 58.012 | 8.804 | 0.917 | 1.000 | 19 | 0.017 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder | af_like | 921.580 | 927.296 | 5.716 | 0.893 | 1.000 | 13 | 0.000 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_guarded | af_like | 531.912 | 555.476 | 23.564 | 0.828 | 1.000 | 57 | 0.041 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_guarded | af_like | 1013.256 | 1068.152 | 54.896 | 0.819 | 1.000 | 132 | 0.291 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_fpaware | af_like | 533.012 | 555.476 | 22.464 | 0.817 | 1.000 | 55 | 0.038 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_guarded | af_like | 49.208 | 57.440 | 8.232 | 0.795 | 1.000 | 18 | 0.016 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_fpaware | af_like | 1013.256 | 1068.152 | 54.896 | 0.793 | 1.000 | 132 | 0.291 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_fpaware | af_like | 50.548 | 57.440 | 6.892 | 0.787 | 1.000 | 16 | 0.014 | beat_state_scoring |
| mitdb/201 | pasm_ml_decoder | ectopic_like | 545.075 | 549.489 | 4.414 | 0.781 | 1.000 | 5 | 0.000 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_guarded | af_like | 921.580 | 927.296 | 5.716 | 0.757 | 1.000 | 13 | 0.000 | beat_state_scoring |
| mitdb/203 | pasm_physionet | ectopic_like | 728.292 | 729.061 | 0.769 | 0.551 | 1.000 | 3 | 0.000 | decoder_short_episode |

## Interpretation

- `pasm_physionet` is the deterministic PASM baseline with PhysioNet evidence postprocessing.
- `pasm_ml_decoder` is the first learned PASM scorer: softmax regression on PASM feature tables.
- `pasm_ml_decoder_guarded` adds normal bias and conservative episode filters to reduce false positives.
- `pasm_ml_decoder_fpaware` retrains with capped class weights and hard-negative normal beats from train false positives.
- If learned variants underperform the baseline, keep them as experimental scaffolds and expand patient-wise data before moving to TCN/Transformer.
