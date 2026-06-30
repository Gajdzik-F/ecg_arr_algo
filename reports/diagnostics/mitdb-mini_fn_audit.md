# False-Negative Stage Audit

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
