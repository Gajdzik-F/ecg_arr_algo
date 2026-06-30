# PASM-Rhythm ML Validation

This report evaluates lightweight NumPy softmax decoders and a candidate-level PASM-AI episode reranker.
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
| ectopic_like | 33 |
| normal | 5985 |

Holdout:

| label | beats |
| --- | --- |
| af_like | 1877 |
| ectopic_like | 74 |
| normal | 3117 |

## Tuned Raw ML Decoder Thresholds

| state | min_confidence |
| --- | --- |
| af_like | 0.360 |
| ectopic_like | 0.280 |
| noise_uncertain | 0.300 |
| sinus_brady | 0.240 |
| sinus_tachy | 0.240 |

## Tuned FP-Aware Decoder Thresholds

| state | min_confidence |
| --- | --- |
| af_like | 0.360 |
| ectopic_like | 0.280 |
| noise_uncertain | 0.300 |
| sinus_brady | 0.240 |
| sinus_tachy | 0.240 |

## Hard Negative Training

| hard_negative_beats | hard_negative_boost |
| --- | --- |
| 64.000 | 3.000 |

## Tuned Guard Config

| parameter | value |
| --- | --- |
| normal_bias | 0.100 |
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

## PASM-AI Episode Reranker

The reranker is a lightweight NumPy logistic model trained on candidate episodes labelled by episode IoU.
`safe` keeps relaxed ectopy diagnostic-only; `v2` uses pattern-aware relaxed rescue.

| model | parameter | value |
| --- | --- | --- |
| pasm_ai_reranker_safe | accept_threshold | 0.300 |
| pasm_ai_reranker_v2 | accept_threshold | 0.700 |
| pasm_ai_reranker_v2 | pause_min_pause_support | 0.350 |
| pasm_ai_reranker_v2 | relaxed_min_proba | 0.900 |
| pasm_ai_reranker_v2 | short_coupled_min_rr_support | 0.260 |

Episode hard negatives:

| hard_negative_candidates | hard_negative_boost |
| --- | --- |
| 17.000 | 3.000 |

Candidate labels:

| split | accepted | candidates |
| --- | --- | --- |
| holdout | 0 | 639 |
| holdout | 1 | 24 |
| train | 0 | 424 |
| train | 1 | 27 |

Accepted candidates by source:

| split | source | pattern | accepted | candidates |
| --- | --- | --- | --- | --- |
| holdout | baseline | baseline | 1 | 3 |
| holdout | baseline | short_coupled_run | 0 | 9 |
| holdout | baseline | short_coupled_run | 1 | 6 |
| holdout | relaxed_ectopy | morphology_cluster | 0 | 227 |
| holdout | relaxed_ectopy | morphology_cluster | 1 | 4 |
| holdout | relaxed_ectopy | premature_plus_pause | 0 | 166 |
| holdout | relaxed_ectopy | short_coupled_run | 0 | 127 |
| holdout | relaxed_ectopy | short_coupled_run | 1 | 11 |
| holdout | state_score | state_score_segment | 0 | 110 |
| train | baseline | baseline | 1 | 3 |
| train | baseline | short_coupled_run | 1 | 3 |
| train | relaxed_ectopy | morphology_cluster | 0 | 310 |
| train | relaxed_ectopy | morphology_cluster | 1 | 19 |
| train | relaxed_ectopy | premature_plus_pause | 0 | 66 |
| train | relaxed_ectopy | premature_plus_pause | 1 | 1 |
| train | relaxed_ectopy | short_coupled_run | 0 | 6 |
| train | relaxed_ectopy | short_coupled_run | 1 | 1 |
| train | state_score | state_score_segment | 0 | 42 |

Uncovered truth episodes:

| split | type | uncovered_truth |
| --- | --- | --- |
| holdout | ectopic_like | 1 |
| train | af_like | 1 |

Top coefficients:

| feature | coefficient | abs_coefficient |
| --- | --- | --- |
| mean_rr_prev | -0.190 | 0.190 |
| mean_morph_z | 0.169 | 0.169 |
| rr_support | 0.160 | 0.160 |
| rr_pause_product | -0.146 | 0.146 |
| local_cv | 0.141 | 0.141 |
| max_state_score | -0.121 | 0.121 |
| min_rr_prev | -0.116 | 0.116 |
| beats | 0.110 | 0.110 |
| baseline_candidate_flag | 0.110 | 0.110 |
| pattern_baseline | 0.108 | 0.108 |
| confidence | -0.107 | 0.107 |
| short_episode_flag | -0.093 | 0.093 |

AI rescued vs rejected:

| metric | count |
| --- | --- |
| baseline_tp | 9 |
| relaxed_tp_rescued | 0 |
| relaxed_candidates_rejected | 535 |
| uncovered_truth | 1 |

MITDB leave-one-record-out reranker v2:

| test_record_id | train_record_ids | threshold | f1 | precision | recall | false_alarms_per_hour |
| --- | --- | --- | --- | --- | --- | --- |
| mitdb/200 | mitdb/205,mitdb/201,mitdb/203 | 0.700 | 0.667 | 1.000 | 0.500 | 0.000 |
| mitdb/205 | mitdb/200,mitdb/201,mitdb/203 | 0.700 | 1.000 | 1.000 | 1.000 | 0.000 |
| mitdb/201 | mitdb/200,mitdb/205,mitdb/203 | 0.700 | 0.000 | 0.000 | 0.000 | 12.000 |
| mitdb/203 | mitdb/200,mitdb/205,mitdb/201 | 0.700 | 0.500 | 0.500 | 0.500 | 24.000 |

## Train Summary

| model | episode_f1_mean | episode_precision_mean | episode_recall_mean | false_alarms_per_hour_mean | typed_f1_mean |
| --- | --- | --- | --- | --- | --- |
| pasm_ai_reranker_safe | 0.867 | 1.000 | 0.792 | 0.000 | 0.973 |
| pasm_ai_reranker_v2 | 0.867 | 1.000 | 0.792 | 0.000 | 0.973 |
| pasm_physionet | 0.867 | 1.000 | 0.792 | 0.000 | 0.973 |
| pasm_ml_decoder | 0.458 | 0.604 | 0.667 | 6.250 | 0.892 |
| pasm_ml_decoder_guarded | 0.424 | 0.688 | 0.542 | 5.250 | 0.885 |
| pasm_ml_decoder_fpaware | 0.392 | 0.655 | 0.542 | 5.250 | 0.878 |

## Holdout Summary

| model | episode_f1_mean | episode_precision_mean | episode_recall_mean | false_alarms_per_hour_mean | typed_f1_mean |
| --- | --- | --- | --- | --- | --- |
| pasm_ai_reranker_safe | 0.500 | 0.500 | 0.500 | 12.000 | 0.900 |
| pasm_ai_reranker_v2 | 0.500 | 0.500 | 0.500 | 12.000 | 0.900 |
| pasm_physionet | 0.500 | 0.500 | 0.500 | 12.000 | 0.900 |
| pasm_ml_decoder | 0.286 | 0.583 | 0.333 | 5.000 | 0.857 |
| pasm_ml_decoder_fpaware | 0.250 | 0.867 | 0.333 | 2.000 | 0.850 |
| pasm_ml_decoder_guarded | 0.250 | 0.867 | 0.333 | 2.000 | 0.850 |

## Holdout Per-Record Metrics

| record_id | model | tp | fp | fn | precision | recall | f1 | false_alarms_per_hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afdb/04126 | pasm_ai_reranker_safe | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| afdb/04126 | pasm_ai_reranker_v2 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| afdb/04126 | pasm_ml_decoder | 3 | 1 | 0 | 0.750 | 1.000 | 0.857 | 3.000 |
| afdb/04126 | pasm_ml_decoder_fpaware | 3 | 2 | 0 | 0.600 | 1.000 | 0.750 | 6.000 |
| afdb/04126 | pasm_ml_decoder_guarded | 3 | 2 | 0 | 0.600 | 1.000 | 0.750 | 6.000 |
| afdb/04126 | pasm_physionet | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| mitdb/201 | pasm_ai_reranker_safe | 0 | 3 | 1 | 0.000 | 0.000 | 0.000 | 12.000 |
| mitdb/201 | pasm_ai_reranker_v2 | 0 | 3 | 1 | 0.000 | 0.000 | 0.000 | 12.000 |
| mitdb/201 | pasm_ml_decoder | 0 | 3 | 1 | 0.000 | 0.000 | 0.000 | 12.000 |
| mitdb/201 | pasm_ml_decoder_fpaware | 0 | 0 | 1 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/201 | pasm_ml_decoder_guarded | 0 | 0 | 1 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/201 | pasm_physionet | 0 | 3 | 1 | 0.000 | 0.000 | 0.000 | 12.000 |
| mitdb/203 | pasm_ai_reranker_safe | 6 | 6 | 6 | 0.500 | 0.500 | 0.500 | 24.000 |
| mitdb/203 | pasm_ai_reranker_v2 | 6 | 6 | 6 | 0.500 | 0.500 | 0.500 | 24.000 |
| mitdb/203 | pasm_ml_decoder | 0 | 0 | 12 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/203 | pasm_ml_decoder_fpaware | 0 | 0 | 12 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/203 | pasm_ml_decoder_guarded | 0 | 0 | 12 | 1.000 | 0.000 | 0.000 | 0.000 |
| mitdb/203 | pasm_physionet | 6 | 6 | 6 | 0.500 | 0.500 | 0.500 | 24.000 |

## Holdout FP/h By Record

| record_id | model | false_alarms_per_hour |
| --- | --- | --- |
| afdb/04126 | pasm_ai_reranker_safe | 0.000 |
| afdb/04126 | pasm_ai_reranker_v2 | 0.000 |
| afdb/04126 | pasm_ml_decoder | 3.000 |
| afdb/04126 | pasm_ml_decoder_fpaware | 6.000 |
| afdb/04126 | pasm_ml_decoder_guarded | 6.000 |
| afdb/04126 | pasm_physionet | 0.000 |
| mitdb/201 | pasm_ai_reranker_safe | 12.000 |
| mitdb/201 | pasm_ai_reranker_v2 | 12.000 |
| mitdb/201 | pasm_ml_decoder | 12.000 |
| mitdb/201 | pasm_ml_decoder_fpaware | 0.000 |
| mitdb/201 | pasm_ml_decoder_guarded | 0.000 |
| mitdb/201 | pasm_physionet | 12.000 |
| mitdb/203 | pasm_ai_reranker_safe | 24.000 |
| mitdb/203 | pasm_ai_reranker_v2 | 24.000 |
| mitdb/203 | pasm_ml_decoder | 0.000 |
| mitdb/203 | pasm_ml_decoder_fpaware | 0.000 |
| mitdb/203 | pasm_ml_decoder_guarded | 0.000 |
| mitdb/203 | pasm_physionet | 24.000 |

## Holdout False Positives By Type

| model | type | false_positives |
| --- | --- | --- |
| pasm_ai_reranker_safe | ectopic_like | 9 |
| pasm_ai_reranker_v2 | ectopic_like | 9 |
| pasm_ml_decoder | ectopic_like | 3 |
| pasm_ml_decoder | af_like | 1 |
| pasm_ml_decoder_fpaware | af_like | 2 |
| pasm_ml_decoder_guarded | af_like | 2 |
| pasm_physionet | ectopic_like | 9 |

## Holdout False Positives By Failure Stage

| model | failure_stage | false_positives |
| --- | --- | --- |
| pasm_ai_reranker_safe | decoder_short_episode | 9 |
| pasm_ai_reranker_v2 | decoder_short_episode | 9 |
| pasm_ml_decoder | decoder_low_confidence | 3 |
| pasm_ml_decoder | beat_state_scoring | 1 |
| pasm_ml_decoder_fpaware | beat_state_scoring | 2 |
| pasm_ml_decoder_guarded | beat_state_scoring | 2 |
| pasm_physionet | decoder_low_confidence | 8 |
| pasm_physionet | decoder_short_episode | 1 |

## FP Removed By Guard Reason

| model | type | reason | removed |
| --- | --- | --- | --- |
| pasm_ml_decoder_fpaware | af_like | low_confidence+too_few_beats | 8 |
| pasm_ml_decoder_fpaware | af_like | low_confidence | 6 |
| pasm_ml_decoder_fpaware | af_like | too_few_beats | 3 |
| pasm_ml_decoder_fpaware | ectopic_like | low_confidence | 1 |
| pasm_ml_decoder_fpaware | ectopic_like | low_confidence+too_few_beats | 1 |
| pasm_ml_decoder_guarded | af_like | low_confidence+too_few_beats | 7 |
| pasm_ml_decoder_guarded | af_like | too_few_beats | 6 |
| pasm_ml_decoder_guarded | af_like | low_confidence | 4 |
| pasm_ml_decoder_guarded | ectopic_like | low_confidence | 1 |
| pasm_ml_decoder_guarded | ectopic_like | low_confidence+too_few_beats | 1 |

## FP Removed By PASM-AI

| model | type | source | pattern | reason | removed |
| --- | --- | --- | --- | --- | --- |
| pasm_ai_reranker_safe | ectopic_like | relaxed_ectopy | morphology_cluster | below_ai_threshold | 231 |
| pasm_ai_reranker_safe | ectopic_like | relaxed_ectopy | premature_plus_pause | below_ai_threshold | 166 |
| pasm_ai_reranker_safe | ectopic_like | relaxed_ectopy | short_coupled_run | below_ai_threshold | 138 |
| pasm_ai_reranker_safe | sinus_tachy | state_score | state_score_segment | below_ai_threshold | 110 |
| pasm_ai_reranker_v2 | ectopic_like | relaxed_ectopy | morphology_cluster | below_ai_threshold | 231 |
| pasm_ai_reranker_v2 | ectopic_like | relaxed_ectopy | premature_plus_pause | below_ai_threshold | 166 |
| pasm_ai_reranker_v2 | ectopic_like | relaxed_ectopy | short_coupled_run | below_ai_threshold | 138 |
| pasm_ai_reranker_v2 | sinus_tachy | state_score | state_score_segment | below_ai_threshold | 110 |

## Top Holdout False-Positive Episodes

| record_id | model | type | start_s | end_s | duration_s | confidence | mean_sqi | beats | best_iou | failure_stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 716.714 | 717.481 | 0.767 | 0.945 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 728.292 | 729.061 | 0.769 | 0.943 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 747.828 | 748.486 | 0.658 | 0.931 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 764.878 | 765.525 | 0.647 | 0.924 | 1.000 | 3 | 0.000 | decoder_short_episode |
| afdb/04126 | pasm_ml_decoder | af_like | 458.004 | 556.984 | 98.980 | 0.897 | 1.000 | 234 | 0.185 | beat_state_scoring |
| mitdb/201 | pasm_ai_reranker_safe | ectopic_like | 211.322 | 212.325 | 1.003 | 0.885 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 796.303 | 797.244 | 0.942 | 0.878 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/201 | pasm_ai_reranker_safe | ectopic_like | 48.403 | 49.375 | 0.972 | 0.867 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 309.356 | 310.394 | 1.039 | 0.838 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/201 | pasm_ai_reranker_safe | ectopic_like | 118.233 | 119.506 | 1.272 | 0.831 | 1.000 | 3 | 0.000 | decoder_short_episode |
| afdb/04126 | pasm_ml_decoder_guarded | af_like | 531.572 | 556.984 | 25.412 | 0.820 | 1.000 | 60 | 0.041 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_guarded | af_like | 458.004 | 529.684 | 71.680 | 0.819 | 1.000 | 172 | 0.141 | beat_state_scoring |

## Interpretation

- `pasm_physionet` is the deterministic PASM baseline with PhysioNet evidence postprocessing.
- `pasm_ml_decoder` is the first learned PASM scorer: softmax regression on PASM feature tables.
- `pasm_ml_decoder_guarded` adds normal bias and conservative episode filters to reduce false positives.
- `pasm_ml_decoder_fpaware` retrains with capped class weights and hard-negative normal beats from train false positives.
- `pasm_ai_reranker_safe` is the conservative candidate-level explainable AI layer.
- `pasm_ai_reranker_v2` adds pattern-aware relaxed ectopy rescue with episode-level hard negatives.
- If learned variants underperform the baseline, keep them as experimental scaffolds and expand patient-wise data before moving to TCN/Transformer.
