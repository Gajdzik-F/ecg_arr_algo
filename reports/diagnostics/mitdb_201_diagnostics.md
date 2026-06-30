# Diagnostics: mitdb/201

## Status Counts

| status | count |
| --- | --- |
| FN | 1 |
| FP | 3 |

## Pattern Counts

| pattern | count |
| --- | --- |
| short_coupled_run | 3 |

## Episodes

| status | start_s | end_s | duration_s | type | confidence | beats | candidate_density | candidate_rate_per_hour | pattern | rr_support | pause_support | morph_support | density_support | mean_rr_prev | min_rr_prev | max_rr_prev | mean_rr_next | local_cv | local_rmssd | mean_morph_z | reason | context_rr_prev | context_rr_next | context_local_cv | context_local_rmssd | matched_truth_index | iou |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP | 48.403 | 49.375 | 0.972 | ectopic_like | 0.433 | 3.000 | 1.000 | 40.075 | short_coupled_run | 0.289 | 0.000 | 1.000 | 0.167 | 0.467 | 0.356 | 0.617 | 0.494 | 0.189 | 0.154 | 1.029 | short_coupled_run ectopic evidence | 0.428 | 0.356 | 0.155 | 0.145 |  | 0.000 |
| FP | 118.233 | 119.506 | 1.272 | ectopic_like | 0.417 | 3.000 | 1.000 | 40.075 | short_coupled_run | 0.244 | 0.410 | 1.000 | 0.167 | 0.562 | 0.378 | 0.894 | 0.628 | 0.206 | 0.178 | 0.739 | short_coupled_run ectopic evidence | 0.414 | 0.378 | 0.183 | 0.168 |  | 0.000 |
| FP | 211.322 | 212.325 | 1.003 | ectopic_like | 0.461 | 3.000 | 1.000 | 40.075 | short_coupled_run | 0.250 | 0.192 | 1.000 | 0.167 | 0.459 | 0.375 | 0.608 | 0.614 | 0.318 | 0.258 | 0.623 | short_coupled_run ectopic evidence | 0.375 | 0.394 | 0.299 | 0.279 |  | 0.000 |
| FN | 477.453 | 478.431 | 0.978 | ectopic_like |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.064 | 0.503 | 0.202 | 0.377 | 0 | 0.000 |
