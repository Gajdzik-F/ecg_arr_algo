# PASM-Rhythm ML Validation

This report evaluates lightweight NumPy softmax decoders and a candidate-level PASM-AI episode reranker.
It is a research checkpoint, not clinical validation.

Preset: `tiny`

## Patient-Wise Split

Requested train records:

| record_id |
| --- |
| mitdb/200 |
| mitdb/205 |
| afdb/04015 |
| afdb/04043 |

Requested holdout records:

| record_id |
| --- |
| mitdb/201 |
| mitdb/203 |
| afdb/04126 |

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

_No rows._

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
| ectopic_like | 0.300 |
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
| 65.000 | 3.000 |

## Tuned Guard Config

| parameter | value |
| --- | --- |
| normal_bias | 0.100 |
| min_episode_confidence | 0.500 |
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
| holdout | 0 | 1446 |
| holdout | 1 | 35 |
| train | 0 | 467 |
| train | 1 | 31 |

Accepted candidates by source:

| split | source | pattern | accepted | candidates |
| --- | --- | --- | --- | --- |
| holdout | baseline | baseline | 1 | 3 |
| holdout | baseline | short_coupled_run | 0 | 9 |
| holdout | baseline | short_coupled_run | 1 | 6 |
| holdout | relaxed_ectopy | morphology_cluster | 0 | 227 |
| holdout | relaxed_ectopy | morphology_cluster | 1 | 4 |
| holdout | relaxed_ectopy | premature_plus_pause | 0 | 166 |
| holdout | relaxed_ectopy | rr_irregular_burst | 0 | 807 |
| holdout | relaxed_ectopy | rr_irregular_burst | 1 | 11 |
| holdout | relaxed_ectopy | short_coupled_run | 0 | 127 |
| holdout | relaxed_ectopy | short_coupled_run | 1 | 11 |
| holdout | state_score | state_score_segment | 0 | 110 |
| train | baseline | baseline | 1 | 3 |
| train | baseline | short_coupled_run | 1 | 3 |
| train | relaxed_ectopy | morphology_cluster | 0 | 310 |
| train | relaxed_ectopy | morphology_cluster | 1 | 19 |
| train | relaxed_ectopy | premature_plus_pause | 0 | 66 |
| train | relaxed_ectopy | premature_plus_pause | 1 | 1 |
| train | relaxed_ectopy | rr_irregular_burst | 0 | 43 |
| train | relaxed_ectopy | rr_irregular_burst | 1 | 4 |
| train | relaxed_ectopy | short_coupled_run | 0 | 6 |
| train | relaxed_ectopy | short_coupled_run | 1 | 1 |
| train | state_score | state_score_segment | 0 | 42 |

Uncovered truth episodes:

| split | type | uncovered_truth |
| --- | --- | --- |
| train | af_like | 1 |

Top coefficients:

| feature | coefficient | abs_coefficient |
| --- | --- | --- |
| mean_rr_prev | -0.236 | 0.236 |
| mean_morph_z | 0.205 | 0.205 |
| rr_support | 0.184 | 0.184 |
| min_rr_prev | -0.139 | 0.139 |
| local_cv | 0.126 | 0.126 |
| max_state_score | -0.116 | 0.116 |
| confidence | -0.113 | 0.113 |
| beats | 0.113 | 0.113 |
| baseline_candidate_flag | 0.111 | 0.111 |
| pattern_baseline | 0.109 | 0.109 |
| rr_pause_product | -0.104 | 0.104 |
| short_episode_flag | -0.093 | 0.093 |

AI rescued vs rejected:

| metric | count |
| --- | --- |
| baseline_tp | 9 |
| relaxed_tp_rescued | 0 |
| relaxed_candidates_rejected | 1353 |
| uncovered_truth | 0 |

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
| pasm_ml_decoder | 0.416 | 0.562 | 0.667 | 7.000 | 0.883 |
| pasm_ml_decoder_guarded | 0.416 | 0.562 | 0.667 | 7.000 | 0.883 |
| pasm_ml_decoder_fpaware | 0.408 | 0.556 | 0.667 | 7.750 | 0.882 |

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
| pasm_ml_decoder_fpaware | af_like | too_few_beats | 13 |
| pasm_ml_decoder_fpaware | ectopic_like | low_confidence | 2 |
| pasm_ml_decoder_fpaware | ectopic_like | low_confidence+too_few_beats | 2 |
| pasm_ml_decoder_guarded | af_like | too_few_beats | 15 |
| pasm_ml_decoder_guarded | ectopic_like | low_confidence | 1 |
| pasm_ml_decoder_guarded | ectopic_like | low_confidence+too_few_beats | 1 |

## FP Removed By PASM-AI

| model | type | source | pattern | reason | removed |
| --- | --- | --- | --- | --- | --- |
| pasm_ai_reranker_safe | ectopic_like | relaxed_ectopy | rr_irregular_burst | below_ai_threshold | 818 |
| pasm_ai_reranker_safe | ectopic_like | relaxed_ectopy | morphology_cluster | below_ai_threshold | 231 |
| pasm_ai_reranker_safe | ectopic_like | relaxed_ectopy | premature_plus_pause | below_ai_threshold | 166 |
| pasm_ai_reranker_safe | ectopic_like | relaxed_ectopy | short_coupled_run | below_ai_threshold | 138 |
| pasm_ai_reranker_safe | sinus_tachy | state_score | state_score_segment | below_ai_threshold | 110 |
| pasm_ai_reranker_v2 | ectopic_like | relaxed_ectopy | rr_irregular_burst | below_ai_threshold | 818 |
| pasm_ai_reranker_v2 | ectopic_like | relaxed_ectopy | morphology_cluster | below_ai_threshold | 231 |
| pasm_ai_reranker_v2 | ectopic_like | relaxed_ectopy | premature_plus_pause | below_ai_threshold | 166 |
| pasm_ai_reranker_v2 | ectopic_like | relaxed_ectopy | short_coupled_run | below_ai_threshold | 138 |
| pasm_ai_reranker_v2 | sinus_tachy | state_score | state_score_segment | below_ai_threshold | 110 |

## Top Holdout False-Positive Episodes

| record_id | model | type | start_s | end_s | duration_s | confidence | mean_sqi | beats | best_iou | failure_stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 716.714 | 717.481 | 0.767 | 0.951 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 728.292 | 729.061 | 0.769 | 0.950 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 747.828 | 748.486 | 0.658 | 0.937 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 764.878 | 765.525 | 0.647 | 0.933 | 1.000 | 3 | 0.000 | decoder_short_episode |
| afdb/04126 | pasm_ml_decoder | af_like | 458.004 | 556.984 | 98.980 | 0.910 | 1.000 | 234 | 0.185 | beat_state_scoring |
| mitdb/201 | pasm_ai_reranker_safe | ectopic_like | 211.322 | 212.325 | 1.003 | 0.890 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/203 | pasm_ai_reranker_safe | ectopic_like | 796.303 | 797.244 | 0.942 | 0.887 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/201 | pasm_ai_reranker_safe | ectopic_like | 48.403 | 49.375 | 0.972 | 0.882 | 1.000 | 3 | 0.000 | decoder_short_episode |
| mitdb/201 | pasm_ai_reranker_safe | ectopic_like | 118.233 | 119.506 | 1.272 | 0.840 | 1.000 | 3 | 0.000 | decoder_short_episode |
| afdb/04126 | pasm_ml_decoder_guarded | af_like | 458.004 | 529.684 | 71.680 | 0.832 | 1.000 | 172 | 0.141 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_fpaware | af_like | 458.004 | 529.684 | 71.680 | 0.824 | 1.000 | 172 | 0.141 | beat_state_scoring |
| afdb/04126 | pasm_ml_decoder_guarded | af_like | 531.044 | 556.984 | 25.940 | 0.823 | 1.000 | 61 | 0.042 | beat_state_scoring |

## Interpretation

- `pasm_physionet` is the deterministic PASM baseline with PhysioNet evidence postprocessing.
- `pasm_ml_decoder` is the first learned PASM scorer: softmax regression on PASM feature tables.
- `pasm_ml_decoder_guarded` adds normal bias and conservative episode filters to reduce false positives.
- `pasm_ml_decoder_fpaware` retrains with capped class weights and hard-negative normal beats from train false positives.
- `pasm_ai_reranker_safe` is the conservative candidate-level explainable AI layer.
- `pasm_ai_reranker_v2` adds pattern-aware relaxed ectopy rescue with episode-level hard negatives.
- If learned variants underperform the baseline, keep them as experimental scaffolds and expand patient-wise data before moving to TCN/Transformer.
