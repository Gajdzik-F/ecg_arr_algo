# ECG Arrhythmia Toolkit - PASM-only code walkthrough

Ten dokument opisuje aktualny kod po usunieciu starej sciezki 1D CNN +
clustering. Jedynym algorytmem detekcji arytmii w repozytorium jest teraz
PASM-Rhythm oraz jego warstwy walidacyjne dla danych syntetycznych i PhysioNet.

Projekt pozostaje prototypem badawczym. Nie jest narzedziem klinicznym.

## 1. Cel

Kod sluzy do badania arytmii w zapisach EKG przez:

- wczytanie sygnalu z CSV albo rekordow PhysioNet przez `wfdb`,
- filtracje, normalizacje i estymacje jakosci sygnalu,
- prace na gotowych R-peakach,
- ekstrakcje beat windows jako informacji morfologicznej,
- uruchomienie PASM-Rhythm,
- budowe per-beat datasetu PASM i lekki learned decoder w NumPy,
- walidacje episode-level,
- zapis raportow Markdown i HTML.

Stara sciezka Beat2Vec/HDBSCAN/DBSCAN zostala usunieta. W kodzie nie ma juz
`beat2vec.py`, `clustering.py`, `episodes.py` ani baseline `legacy_rr`.

## 2. Mapa Plikow

| Plik | Rola |
| --- | --- |
| `ecgio.py` | Wczytywanie CSV i estymacja czestotliwosci probkowania. |
| `preprocess.py` | Resampling, bandpass, notch filter i robust normalization. |
| `sqi.py` | Signal Quality Index dla okien sygnalu. |
| `beats.py` | Ekstrakcja beatow wokol R-peakow i cechy RR. |
| `pasm_rhythm.py` | Rdzen PASM: pamiec pacjenta, cechy rytmu, scoring stanow, graf i dekoder epizodow. |
| `pasm_dataset.py` | Per-beat dataset cech PASM z etykietami z truth episodes. |
| `pasm_ml_decoder.py` | Lekki softmax decoder w NumPy, bez nowych zaleznosci ML. |
| `pasm_ml_validation.py` | Patient-wise benchmark learned, guarded, FP-aware decoder i deterministic `pasm_physionet`. |
| `pasm_validation.py` | Syntetyczna kohorta, strojenie progow PASM i metryki episode-level. |
| `pasm_physionet.py` | Ladowanie MITDB/AFDB, mapowanie adnotacji, evidence layer i diagnostyka. |
| `pasm_realdata.py` | Presety real-data, benchmark laczony, Markdown/HTML i localhost. |
| `pasm_html_report.py` | Statyczne raporty HTML i diagnostyka rekordu. |
| `viz.py` | Proste wykresy R-peakow i SQI. |
| `tests/` | Testy regresyjne rdzenia, walidacji, PhysioNet i raportow. |

## 3. Przeplyw Danych

### CSV

1. `ecgio.load_ecg_csv(...)` wczytuje czas i wartosci sygnalu.
2. `ecgio.estimate_fs_from_time(...)` szacuje `fs`, gdy CSV ma os czasu.
3. `preprocess.bandpass_filter(...)` i `robust_normalize(...)` przygotowuja EKG.
4. `sqi.compute_sqi(...)` liczy jakosc sygnalu.
5. Zewnetrzny detektor R-peakow dostarcza indeksy R.
6. `beats.extract_beats(...)` wycina beat windows.
7. `beats.compute_rr_times(...)` tworzy `r_times`, `rr_prev`, `rr_next`.
8. `pasm_rhythm.run_pasm_rhythm(...)` uruchamia PASM.

### PhysioNet

1. `pasm_physionet.load_mitdb_record(...)` albo `load_afdb_record(...)` pobiera rekord.
2. MITDB jest mapowany na beat-level truth dla `ectopic_like`.
3. AFDB jest mapowany na rhythm-level truth dla `af_like`.
4. `run_pasm_physionet_pipeline(...)` uruchamia preprocessing, beat extraction, PASM i evidence layer.
5. Predykcje sa filtrowane do zakresu adnotacji bazy.
6. `evaluate_physionet_records(...)` liczy metryki episode-level.

### Real-Data Presets

1. `pasm_realdata.load_records_for_preset(...)` wybiera rekordy.
2. `run_realdata_preset(...)` laduje je i uruchamia ewaluacje.
3. `write_realdata_report(...)` zapisuje Markdown.
4. `write_realdata_html_report(...)` zapisuje HTML.
5. `serve_report(...)` moze wystawic raport lokalnie.

### PASM v0.4 ML Benchmark

1. `pasm_ml_validation.py` laduje preset `tiny`, `mini`, `mitdb-mini` albo `afdb-mini`.
2. `run_pasm_physionet_pipeline(...)` generuje cechy, state scores i epizody baseline.
3. `pasm_dataset.build_pasm_feature_frame(...)` tworzy per-beat tabele cech z labelami.
4. `pasm_ml_decoder.fit_softmax_decoder(...)` trenuje softmax regression w NumPy z capped class weights.
5. False-positive epizody na train wskazuja hard-negative normal beats.
6. FP-aware decoder jest trenowany ponownie z podbitymi wagami hard negatives.
7. Learned state scores przechodza przez istniejacy `decode_pasm_episodes(...)`.
8. Guarded decoder dodaje `normal_bias`, konserwatywne filtry epizodow i ectopy-specific support guard.
9. Raport porownuje `pasm_physionet`, `pasm_ml_decoder`, `pasm_ml_decoder_guarded` i `pasm_ml_decoder_fpaware`.
10. Raport pokazuje FP/h per rekord, false positives by type, guard removals i top FP episodes.

## 4. Preprocessing

`preprocess.py` przygotowuje sygnal:

- `resample_to_fs(...)` resampluje sygnal przez `scipy.signal.resample_poly`.
- `bandpass_filter(...)` filtruje typowo w zakresie 0.5-40 Hz.
- `notch_filter(...)` usuwa zaklocenie sieciowe, gdy ma to sens dla `fs`.
- `robust_normalize(...)` skaluje sygnal przez mediane i MAD.

W PhysioNet funkcja `prepare_ecg_for_pasm(...)` laczy bandpass i robust
normalization, a w razie problemu ma fallback do samej normalizacji.

## 5. SQI

`sqi.compute_sqi(...)` dzieli sygnal na okna i liczy m.in.:

- moc pasma QRS,
- moc niskich i wysokich czestotliwosci,
- flatline,
- clipping,
- amplitude range.

Wynikiem jest `sqi` w zakresie 0-1. `sqi_at_times(...)` dopasowuje te wartosci
do czasow beatow. PASM uzywa SQI jako czesci wiarygodnosci i do stanu
`noise_uncertain`.

## 6. Beaty I RR

`beats.extract_beats(...)` wycina okna wokol R-peakow i normalizuje je per beat.
Te beat windows sa uzywane przez PASM jako prosta informacja morfologiczna.

`compute_rr_times(...)` zwraca:

- `r_times`,
- `rr_prev`,
- `rr_next`.

`select_rr_for_beats(...)` dopasowuje RR do beatow, jesli ekstrakcja pominela
beaty przy krawedziach sygnalu.

## 7. Rdzen PASM-Rhythm

Glowny plik: `pasm_rhythm.py`.

Stany PASM:

- `normal`,
- `sinus_tachy`,
- `sinus_brady`,
- `af_like`,
- `ectopic_like`,
- `noise_uncertain`.

### PatientMemory

`PatientMemory` przechowuje baseline pacjenta:

- medianowe RR,
- skale zmiennosci RR,
- medianowe RMSSD,
- medianowe SQI,
- prototyp morfologii beatu,
- liczbe beatow w baseline.

To pozwala oceniac rytm wzgledem konkretnego pacjenta, a nie tylko globalnych
progów.

### compute_rhythm_features

`compute_rhythm_features(...)` tworzy tabele beat-level:

- `time_s`,
- `rr_prev`,
- `rr_next`,
- `hr`,
- `delta_rr`,
- `rr_ratio`,
- `local_rr_median`,
- `local_rmssd`,
- `local_cv`,
- `sqi`,
- `rpeak_uncertainty`,
- `reliability`.

### build_patient_memory

`build_patient_memory(...)` bierze poczatkowy fragment nagrania, wybiera
wiarygodne beaty i buduje baseline RR oraz morfologii.

### score_pasm_states

`score_pasm_states(...)` liczy evidence dla kazdego stanu:

- szybki rytm dla `sinus_tachy`,
- wolny rytm dla `sinus_brady`,
- nieregularnosc RR dla `af_like`,
- skoki RR i odchylenie morfologii dla `ectopic_like`,
- niska jakosc dla `noise_uncertain`.

Wynik jest tabela score/probability-like dla wszystkich stanow.

### build_rhythm_graph

`build_rhythm_graph(...)` buduje graf diagnostyczny:

- beat nodes,
- temporal edges,
- state-likelihood edges.

Graf jest czescia kierunku PASM: zamiast pojedynczych decyzji beat-level mozna
analizowac sekwencje i relacje stanow.

### decode_pasm_episodes

`decode_pasm_episodes(...)`:

1. wybiera najlepszy stan per beat,
2. filtruje po minimalnej pewnosci,
3. grupuje kolejne beaty w epizody,
4. odrzuca epizody za krotkie,
5. dopisuje `confidence`, `beats`, `mean_sqi` i `reason`.

### run_pasm_rhythm

`run_pasm_rhythm(...)` jest glownym entrypointem:

1. liczy cechy rytmu,
2. buduje pamiec pacjenta,
3. liczy score stanow,
4. buduje graf,
5. dekoduje epizody.

## 8. Walidacja Syntetyczna

Plik: `pasm_validation.py`.

`make_synthetic_record(...)` generuje sekwencje RR i beat windows z epizodami:

- `sinus_tachy`,
- `sinus_brady`,
- `af_like`,
- `ectopic_like`,
- `noise_uncertain`.

`evaluate_episodes(...)` porownuje predykcje z truth przez IoU:

- TP, FP, FN,
- precision,
- recall,
- F1,
- false alarms/hour.

`tune_pasm_thresholds(...)` stroi progi dekodera PASM na train.

`run_train_holdout_benchmark(...)` generuje osobne train/holdout i raportuje:

- `pasm_tuned`,
- `pasm_default`.

Nie ma juz baseline `legacy_rr`.

## 9. PhysioNet Evidence Layer

Plik: `pasm_physionet.py`.

Warstwa PhysioNet nie zastępuje rdzenia PASM. Dodaje reguly pomagajace
dopasowac wynik do typu adnotacji w MITDB i AFDB.

### AF Evidence

`detect_fast_irregular_af(...)` szuka fragmentow z:

- wysokim HR,
- podwyzszonym `local_cv`,
- podwyzszonym `local_rmssd`,
- wystarczajaca dlugoscia epizodu.

Tworzy epizody `af_like`, szczegolnie istotne dla AFDB.

### Ectopy Evidence

`detect_short_coupled_ectopy(...)` szuka krotko sprzezonych ektopii:

- absolutnie krotkie RR,
- RR krotkie wzgledem rytmu pacjenta,
- kontekst kolejnego RR,
- opcjonalne wsparcie morfologii,
- flood-control dla rekordow z wieloma kandydatami.

To jest wazne glownie dla MITDB.

### Zakres Adnotacji

`filter_predictions_for_annotation_scope(...)` ogranicza predykcje:

- MITDB: tylko `ectopic_like`,
- AFDB: tylko `af_like`.

Dzieki temu metryka nie karze algorytmu za typy arytmii, ktorych dana sciezka
truth w ogole nie adnotuje.

## 10. Diagnostyka

`diagnose_physionet_record(...)` uruchamia pipeline i dopasowuje truth vs
prediction.

`match_diagnostic_episodes(...)` tworzy tabele TP/FP/FN z kontekstem:

- IoU,
- confidence,
- liczba beatow,
- mean SQI,
- RR context,
- local CV/RMSSD,
- morphology z-score, jesli dostepne.

Raport diagnostyczny HTML pozwala szybko zobaczyc, czy bledy sa izolowane,
seryjne, czy przesuniete wzgledem truth.

## 11. Raporty HTML

`pasm_html_report.py` generuje:

- raport real-data preset,
- raport diagnostyczny jednego rekordu,
- timeline truth vs prediction,
- tabele per-record i per-type,
- parametry evidence layer.

Raporty sa statyczne i moga byc otwierane z pliku albo wystawione przez lokalny
serwer.

## 12. CLI Real-Data

`pasm_realdata.py` obsluguje:

- `--preset`,
- `--out`,
- `--html-out`,
- `--serve`,
- `--port`,
- `--diagnose-record`,
- `--diagnostic-html-out`.

Glowne presety:

- `smoke`,
- `mini`,
- `mitdb-smoke`,
- `mitdb-mini`,
- `afdb-smoke`,
- `afdb-mini`.

## 13. Testy

`tests/test_pasm_rhythm.py` sprawdza rdzen PASM.

`tests/test_pasm_validation.py` sprawdza:

- IoU matching,
- syntetyczny benchmark PASM-only,
- train/holdout threshold tuning.

`tests/test_pasm_physionet.py` sprawdza:

- mapowanie MITDB,
- mapowanie AFDB,
- AF evidence,
- ektopie,
- flood-control,
- postprocessing,
- diagnostyke,
- annotation scope.

`tests/test_pasm_realdata.py` sprawdza:

- presety,
- raport Markdown,
- raport HTML,
- diagnostyczny HTML.

## 14. Najczestsze Komendy

Testy:

```powershell
py -3.11 -m unittest discover -s tests -v
```

Syntetyczny train/holdout:

```powershell
py -3.11 pasm_validation.py --train-records 30 --holdout-records 30 --seed 2026 --out PASM_VALIDATION.md
```

Real-data mini:

```powershell
py -3.11 pasm_realdata.py --preset mini --out PASM_REALDATA_SUMMARY.md --html-out reports/pasm_mini.html
```

Diagnostyka MITDB/203:

```powershell
py -3.11 pasm_realdata.py --preset mini --out PASM_REALDATA_SUMMARY.md --diagnose-record mitdb/203 --diagnostic-html-out reports/mitdb_203_diagnostic.html
```

## 15. Kierunek Rozwoju

Najbardziej sensowne kolejne kroki:

1. Rozszerzyc real-data benchmark do patient-wise train/holdout.
2. Zbudowac tabelaryczny beat/window dataset dla MITDB/AFDB.
3. Dodac lekki PASM-compatible scorer ektopii zamiast wracac do klastrowania.
4. Utrzymac syntetyczny benchmark jako regresje.
5. Rozwijac diagnostyczne HTML, bo pokazuje zrodla TP/FP/FN.
