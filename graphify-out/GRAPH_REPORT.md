# Graph Report - C:\Users\fryde\Desktop\ECG_PASM\ecg_arr_algo  (2026-06-28)

## Corpus Check
- Corpus is ~30,793 words - fits in a single context window. You may not need a graph.

## Summary
- 268 nodes · 561 edges · 13 communities
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 14 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Beat PhysioNet Pipeline|Beat PhysioNet Pipeline]]
- [[_COMMUNITY_ML Validation Diagnostics|ML Validation Diagnostics]]
- [[_COMMUNITY_Dataset Decoder Training|Dataset Decoder Training]]
- [[_COMMUNITY_HTML Report Rendering|HTML Report Rendering]]
- [[_COMMUNITY_PASM Concept Evidence|PASM Concept Evidence]]
- [[_COMMUNITY_PASM Rhythm Core|PASM Rhythm Core]]
- [[_COMMUNITY_Synthetic Validation|Synthetic Validation]]
- [[_COMMUNITY_Real Data Next Steps|Real Data Next Steps]]
- [[_COMMUNITY_Codebase Walkthrough|Codebase Walkthrough]]
- [[_COMMUNITY_Signal Quality Index|Signal Quality Index]]
- [[_COMMUNITY_CSV ECG Loading|CSV ECG Loading]]
- [[_COMMUNITY_ML Benchmark Variants|ML Benchmark Variants]]

## God Nodes (most connected - your core abstractions)
1. `run_ml_validation()` - 17 edges
2. `PASMPhysioNetTest` - 15 edges
3. `run_pasm_physionet_pipeline()` - 14 edges
4. `summarize_benchmark()` - 14 edges
5. `run_pasm_rhythm()` - 13 edges
6. `PASMMLTest` - 13 edges
7. `build_pasm_feature_frame()` - 12 edges
8. `write_realdata_html_report()` - 12 edges
9. `fit_softmax_decoder()` - 11 edges
10. `write_ml_validation_report()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `PASM-Rhythm Prototype` --semantically_similar_to--> `Patient-Adaptive State-Memory Rhythm Graph`  [INFERRED] [semantically similar]
  README.md → PASM-Rhythm.md
- `Next Validation Stage Uses Patient-wise Real ECG` --semantically_similar_to--> `Expand Real-data Benchmark Priority`  [INFERRED] [semantically similar]
  PASM_VALIDATION.md → PASM_NEXT_STEPS.md
- `PASM-Rhythm ML Validation Mini` --semantically_similar_to--> `PASM-Rhythm ML Validation Tiny`  [INFERRED] [semantically similar]
  PASM_ML_BENCHMARK.md → PASM_ML_VALIDATION.md
- `AFDB PhysioNet Smoke Validation` --semantically_similar_to--> `AFDB Mini Validation`  [INFERRED] [semantically similar]
  PASM_AFDB_VALIDATION.md → PASM_AFDB_MINI_VALIDATION.md
- `PASMMLTest` --uses--> `PhysioNetRecord`  [INFERRED]
  tests/test_pasm_ml.py → pasm_physionet.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **PASM Pipeline Flow** — pasm_rhythm_rhythm_features, pasm_rhythm_patient_memory, pasm_rhythm_state_scoring, pasm_rhythm_typed_rhythm_graph, pasm_rhythm_episode_decoder [EXTRACTED 1.00]
- **Validation Report Suite** — pasm_validation_synthetic_validation, pasm_ml_validation_ml_validation, pasm_ml_benchmark_mini_benchmark, pasm_realdata_smoke_smoke_preset, pasm_realdata_summary_mini_preset, pasm_physionet_validation_mitdb_smoke, pasm_afdb_validation_afdb_smoke, pasm_afdb_mini_validation_afdb_mini, pasm_mitdb_mini_validation_mitdb_mini [EXTRACTED 1.00]
- **MITDB AFDB Stress Pattern** — pasm_next_steps_afdb_false_alarm_reduction, pasm_next_steps_mitdb_ectopy_stress_cases, pasm_realdata_summary_mini_preset, pasm_afdb_mini_validation_afdb_mini, pasm_mitdb_mini_validation_mitdb_mini, reports_mitdb_203_diagnostic_mitdb_203_diagnostic_report [INFERRED 0.85]

## Communities (13 total, 0 thin omitted)

### Community 0 - "Beat PhysioNet Pipeline"
Cohesion: 0.06
Nodes (42): extract_beats(), _normalize_beats(), Align RR features to beats retained by `extract_beats`.      Returns RR arrays f, select_rr_for_beats(), evaluate_pipeline_records(), beat_labels_to_episodes(), detect_fast_irregular_af(), detect_short_coupled_ectopy() (+34 more)

### Community 1 - "ML Validation Diagnostics"
Cohesion: 0.13
Nodes (34): _add_guard_reasons(), apply_normal_bias(), attach_episode_support(), build_hard_negative_boosts(), classify_prediction_failure(), collect_guard_removed(), collect_holdout_predictions(), _df_to_markdown() (+26 more)

### Community 2 - "Dataset Decoder Training"
Cohesion: 0.11
Nodes (17): assign_beat_labels(), build_pasm_dataset(), build_pasm_feature_frame(), compute_morph_z(), Build a per-beat PASM feature table for one PhysioNetRecord.      Labels are der, _balanced_sample_weights(), _fit_normalizer(), fit_softmax_decoder() (+9 more)

### Community 3 - "HTML Report Rendering"
Cohesion: 0.19
Nodes (25): build_realdata_tables(), _dataframe_to_table(), _diagnostic_cards(), _ectopy_parameters(), _evidence_parameters(), _fmt(), _header(), _interpretation() (+17 more)

### Community 4 - "PASM Concept Evidence"
Cohesion: 0.09
Nodes (23): MITDB PhysioNet Smoke Validation, Real-data Smoke Preset, Real-data Mini Preset, PASM-Rhythm Document, Duration-aware Episode Decoder, FP-Aware Learned Decoder, Learned State Scorer, Do Not Return to Embedding Clustering (+15 more)

### Community 5 - "PASM Rhythm Core"
Cohesion: 0.15
Nodes (19): build_patient_memory(), build_rhythm_graph(), compute_rhythm_features(), decode_pasm_episodes(), _episode_reason(), _mad(), Estimate a patient-specific normal baseline from high-quality early beats., Produce calibrated-looking state scores for the first PASM prototype.      Score (+11 more)

### Community 6 - "Synthetic Validation"
Cohesion: 0.20
Nodes (17): _df_to_markdown(), evaluate_pasm_on_records(), labels_to_episodes(), main(), make_synthetic_cohort(), make_synthetic_record(), pasm_predict(), Generate one patient-like beat sequence with labeled rhythm episodes. (+9 more)

### Community 7 - "Real Data Next Steps"
Cohesion: 0.17
Nodes (13): AFDB Mini Validation, AFDB PhysioNet Smoke Validation, MITDB Mini Validation, Reduce AFDB False Alarms, MITDB Ectopy Stress Cases, PASM-Rhythm Next Steps, Expand Real-data Benchmark Priority, Safe Progress Definition (+5 more)

### Community 8 - "Codebase Walkthrough"
Cohesion: 0.33
Nodes (7): ECG Arrhythmia Toolkit PASM-only Code Walkthrough, CSV PhysioNet Real-data Data Flow, HTML Reports, PASM-Rhythm Core, PASM-only Architecture, PhysioNet Evidence Layer, Regression Tests

### Community 9 - "Signal Quality Index"
Cohesion: 0.38
Nodes (6): _bandpower(), compute_sqi(), _ramp(), Returns:       sqi_t: time centers of windows       sqi: quality in [0,1] (highe, Interpolate SQI to arbitrary times (e.g., R-peak times)., sqi_at_times()

### Community 10 - "CSV ECG Loading"
Cohesion: 0.40
Nodes (4): estimate_fs_from_time(), load_ecg_csv(), Loads CSV that contains either:       - columns: time, value       - or single c, ndarray

### Community 11 - "ML Benchmark Variants"
Cohesion: 0.40
Nodes (5): PASM-Rhythm ML Validation Mini, Hard Negative Training, PASM-Rhythm ML Validation Tiny, pasm_ml_decoder, pasm_ml_decoder_fpaware

## Knowledge Gaps
- **17 isolated node(s):** `CSV PhysioNet Real-data Data Flow`, `PhysioNet Evidence Layer`, `HTML Reports`, `Regression Tests`, `PASM-Rhythm Document` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PASMMLTest` connect `Dataset Decoder Training` to `Beat PhysioNet Pipeline`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `run_pasm_rhythm()` connect `PASM Rhythm Core` to `Beat PhysioNet Pipeline`, `Synthetic Validation`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Why does `PhysioNetRecord` connect `Beat PhysioNet Pipeline` to `Dataset Decoder Training`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **What connects `Align RR features to beats retained by `extract_beats`.      Returns RR arrays f`, `Loads CSV that contains either:       - columns: time, value       - or single c`, `Build a per-beat PASM feature table for one PhysioNetRecord.      Labels are der` to the rest of the system?**
  _40 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Beat PhysioNet Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.06153846153846154 - nodes in this community are weakly interconnected._
- **Should `ML Validation Diagnostics` be split into smaller, more focused modules?**
  _Cohesion score 0.13109243697478992 - nodes in this community are weakly interconnected._
- **Should `Dataset Decoder Training` be split into smaller, more focused modules?**
  _Cohesion score 0.11491935483870967 - nodes in this community are weakly interconnected._