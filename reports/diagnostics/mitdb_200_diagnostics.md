# Diagnostics: mitdb/200

## Status Counts

| status | count |
| --- | --- |
| FN | 1 |
| TP | 1 |

## Pattern Counts

| pattern | count |
| --- | --- |
| short_coupled_run | 1 |

## Episodes

| status | start_s | end_s | duration_s | type | confidence | beats | candidate_density | candidate_rate_per_hour | pattern | rr_support | pause_support | morph_support | density_support | mean_rr_prev | min_rr_prev | max_rr_prev | mean_rr_next | local_cv | local_rmssd | mean_morph_z | reason | context_rr_prev | context_rr_next | context_local_cv | context_local_rmssd | matched_truth_index | iou |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FN | 109.219 | 110.531 | 1.311 | ectopic_like |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.528 | 0.458 | 0.283 | 0.402 | 0 | 0.000 |
| TP | 630.164 | 631.097 | 0.933 | ectopic_like | 0.122 | 3.000 | 1.000 | 4.005 | short_coupled_run | 0.117 | 1.000 | 1.000 | 0.167 | 0.477 | 0.442 | 0.497 | 0.701 | 0.194 | 0.201 | 2.050 | short_coupled_run ectopic evidence | 0.497 | 0.442 | 0.162 | 0.200 | 1 | 1.000 |
