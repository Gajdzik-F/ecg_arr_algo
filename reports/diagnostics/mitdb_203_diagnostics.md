# Diagnostics: mitdb/203

## Status Counts

| status | count |
| --- | --- |
| FN | 6 |
| FP | 6 |
| TP | 6 |

## Pattern Counts

| pattern | count |
| --- | --- |
| short_coupled_run | 12 |

## Episodes

| status | start_s | end_s | duration_s | type | confidence | beats | candidate_density | candidate_rate_per_hour | pattern | rr_support | pause_support | morph_support | density_support | mean_rr_prev | min_rr_prev | max_rr_prev | mean_rr_next | local_cv | local_rmssd | mean_morph_z | reason | context_rr_prev | context_rr_next | context_local_cv | context_local_rmssd | matched_truth_index | iou |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FN | 46.461 | 47.156 | 0.694 | ectopic_like |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.353 | 0.367 | 0.312 | 0.331 | 0 | 0.000 |
| FN | 92.033 | 92.792 | 0.758 | ectopic_like |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.408 | 0.381 | 0.217 | 0.205 | 1 | 0.000 |
| TP | 105.264 | 105.889 | 0.625 | ectopic_like | 0.512 | 3.000 | 6.000 | 512.546 | short_coupled_run | 0.324 | 0.505 | 1.000 | 1.000 | 0.325 | 0.292 | 0.350 | 0.462 | 0.297 | 0.235 | 0.759 | short_coupled_run ectopic evidence | 0.350 | 0.292 | 0.263 | 0.241 | 2 | 1.000 |
| FN | 269.125 | 269.864 | 0.739 | ectopic_like |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.383 | 0.372 | 0.350 | 0.394 | 3 | 0.000 |
| TP | 301.711 | 302.383 | 0.672 | ectopic_like | 0.248 | 3.000 | 6.000 | 512.546 | short_coupled_run | 0.285 | 0.000 | 1.000 | 1.000 | 0.355 | 0.308 | 0.392 | 0.342 | 0.267 | 0.185 | 0.622 | short_coupled_run ectopic evidence | 0.392 | 0.364 | 0.222 | 0.185 | 4 | 0.656 |
| FN | 303.825 | 306.819 | 2.994 | ectopic_like |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.742 | 0.414 | 0.341 | 0.211 | 5 | 0.000 |
| FP | 309.356 | 310.394 | 1.039 | ectopic_like | 0.087 | 3.000 | 6.000 | 512.546 | short_coupled_run | 0.066 | 0.544 | 1.000 | 1.000 | 0.481 | 0.403 | 0.617 | 0.467 | 0.404 | 0.318 | 0.735 | short_coupled_run ectopic evidence | 0.403 | 0.422 | 0.411 | 0.316 |  | 0.000 |
| FN | 511.097 | 511.803 | 0.706 | ectopic_like |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.433 | 0.356 | 0.386 | 0.359 | 6 | 0.000 |
| TP | 641.328 | 641.972 | 0.644 | ectopic_like | 0.461 | 3.000 | 3.000 | 512.546 | short_coupled_run | 0.317 | 0.000 | 1.000 | 0.500 | 0.338 | 0.294 | 0.369 | 0.342 | 0.301 | 0.213 | 0.640 | short_coupled_run ectopic evidence | 0.369 | 0.294 | 0.275 | 0.218 | 7 | 0.629 |
| TP | 710.150 | 710.794 | 0.644 | ectopic_like | 0.473 | 3.000 | 4.000 | 512.546 | short_coupled_run | 0.279 | 0.249 | 1.000 | 0.667 | 0.331 | 0.311 | 0.347 | 0.406 | 0.353 | 0.261 | 0.727 | short_coupled_run ectopic evidence | 0.347 | 0.311 | 0.309 | 0.267 | 8 | 1.000 |
| FP | 716.714 | 717.481 | 0.767 | ectopic_like | 0.415 | 3.000 | 3.000 | 512.546 | short_coupled_run | 0.388 | 0.083 | 1.000 | 0.500 | 0.395 | 0.264 | 0.503 | 0.463 | 0.311 | 0.302 | 0.665 | short_coupled_run ectopic evidence | 0.419 | 0.264 | 0.266 | 0.298 |  | 0.000 |
| FP | 728.292 | 729.061 | 0.769 | ectopic_like | 0.551 | 3.000 | 3.000 | 512.546 | short_coupled_run | 0.362 | 0.069 | 0.990 | 0.500 | 0.373 | 0.275 | 0.494 | 0.453 | 0.324 | 0.254 | 0.594 | short_coupled_run ectopic evidence | 0.350 | 0.275 | 0.293 | 0.253 |  | 0.000 |
| TP | 741.486 | 742.089 | 0.603 | ectopic_like | 0.609 | 3.000 | 3.000 | 512.546 | short_coupled_run | 0.420 | 0.000 | 1.000 | 0.500 | 0.318 | 0.250 | 0.353 | 0.322 | 0.312 | 0.169 | 0.686 | short_coupled_run ectopic evidence | 0.350 | 0.250 | 0.280 | 0.168 | 9 | 0.624 |
| FP | 747.828 | 748.486 | 0.658 | ectopic_like | 0.461 | 3.000 | 4.000 | 512.546 | short_coupled_run | 0.272 | 0.003 | 1.000 | 0.667 | 0.336 | 0.314 | 0.350 | 0.427 | 0.332 | 0.282 | 0.687 | short_coupled_run ectopic evidence | 0.350 | 0.314 | 0.311 | 0.300 |  | 0.000 |
| FN | 758.111 | 759.103 | 0.992 | ectopic_like |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.369 | 0.347 | 0.324 | 0.269 | 10 | 0.000 |
| FP | 764.878 | 765.525 | 0.647 | ectopic_like | 0.467 | 3.000 | 5.000 | 512.546 | short_coupled_run | 0.285 | 0.000 | 1.000 | 0.833 | 0.333 | 0.308 | 0.353 | 0.356 | 0.266 | 0.204 | 0.779 | short_coupled_run ectopic evidence | 0.353 | 0.308 | 0.227 | 0.216 |  | 0.000 |
| TP | 781.958 | 782.631 | 0.672 | ectopic_like | 0.377 | 3.000 | 7.000 | 512.546 | short_coupled_run | 0.253 | 0.000 | 1.000 | 1.000 | 0.350 | 0.322 | 0.378 | 0.429 | 0.276 | 0.190 | 0.749 | short_coupled_run ectopic evidence | 0.378 | 0.322 | 0.266 | 0.199 | 11 | 1.000 |
| FP | 796.303 | 797.244 | 0.942 | ectopic_like | 0.403 | 3.000 | 6.000 | 512.546 | short_coupled_run | 0.246 | 0.084 | 1.000 | 1.000 | 0.422 | 0.325 | 0.578 | 0.427 | 0.241 | 0.173 | 0.620 | short_coupled_run ectopic evidence | 0.325 | 0.364 | 0.240 | 0.172 |  | 0.000 |
