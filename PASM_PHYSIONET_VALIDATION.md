# PASM-Rhythm PhysioNet Validation

This report uses WFDB/PhysioNet annotations. It is a research validation harness, not clinical certification.

## Summary

         model  episode_f1_mean  episode_precision_mean  episode_recall_mean  false_alarms_per_hour_mean  typed_f1_mean
pasm_physionet         0.666667                     1.0                  0.5                         0.0       0.933333

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

## Raw Metrics

record_id          model                  type  tp  fp  fn  precision  recall       f1  mean_iou
mitdb/200 pasm_physionet           sinus_tachy   0   0   0        1.0     1.0 1.000000       NaN
mitdb/200 pasm_physionet           sinus_brady   0   0   0        1.0     1.0 1.000000       NaN
mitdb/200 pasm_physionet               af_like   0   0   0        1.0     1.0 1.000000       NaN
mitdb/200 pasm_physionet          ectopic_like   1   0   1        1.0     0.5 0.666667       1.0
mitdb/200 pasm_physionet       noise_uncertain   0   0   0        1.0     1.0 1.000000       NaN
mitdb/200 pasm_physionet                 macro   1   0   1        1.0     0.5 0.666667       NaN
mitdb/200 pasm_physionet false_alarms_per_hour   1   0   1        NaN     NaN 0.000000       NaN
