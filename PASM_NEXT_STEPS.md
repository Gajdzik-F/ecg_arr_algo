# PASM-Rhythm Next Steps

Ten plik jest notatka awaryjna na wypadek utraty kontekstu lub skonczenia sie tokenow. Traktuj repo jako zrodlo prawdy i przed dalsza praca uruchom aktualne testy oraz przeczytaj raporty walidacyjne.

## Aktualny Stan

Glowne pliki:

- `pasm_rhythm.py` - rdzen PASM-Rhythm: patient memory, rhythm features, state scoring, graph, episode decoder.
- `pasm_dataset.py` - per-beat dataset cech PASM dla uczenia i walidacji.
- `pasm_ml_decoder.py` - lekki learned state scorer: softmax regression w NumPy.
- `pasm_ml_validation.py` - patient-wise benchmark suite: `pasm_physionet` vs `pasm_ml_decoder` vs `pasm_ml_decoder_guarded` vs `pasm_ml_decoder_fpaware`.
- `pasm_validation.py` - syntetyczna kohorta PASM-only oraz strojenie progow train/holdout.
- `pasm_physionet.py` - opcjonalny harness WFDB/PhysioNet dla MITDB i AFDB.
- `tests/` - testy regresji dla rdzenia, walidacji syntetycznej i mapowania PhysioNet.
- `PASM_VALIDATION.md` - syntetyczny train/holdout report.
- `PASM_ML_VALIDATION.md` - patient-wise learned decoder report, jesli zostal wygenerowany.
- `PASM_ML_BENCHMARK.md` - preset `mini` dla learned decoder benchmark suite, jesli zostal wygenerowany.
- `PASM_REALDATA_SMOKE.md` - laczony smoke report MITDB + AFDB.
- `PASM_REALDATA_SUMMARY.md` - trudniejszy laczony mini report MITDB + AFDB.
- `PASM_PHYSIONET_VALIDATION.md` - MITDB smoke report.
- `PASM_AFDB_VALIDATION.md` - AFDB smoke report.

Aktualne wyniki:

```text
Synthetic holdout:
pasm_tuned    episode F1 0.834 | precision 0.773 | recall 0.917 | false alarms/hour 17.581
pasm_default  episode F1 0.749 | precision 0.733 | recall 0.792 | false alarms/hour 19.256

MITDB 200, first 900 s:
episode F1 0.667 | precision 1.000 | recall 0.500 | false alarms/hour 0.000

AFDB 04015 + 04126, first 900 s:
episode F1 0.900 | precision 1.000 | recall 0.833 | false alarms/hour 0.000

Combined real-data smoke preset:
episode F1 0.822 | precision 1.000 | recall 0.722 | false alarms/hour 0.000

Combined real-data mini preset:
episode F1 0.710 | precision 0.786 | recall 0.667 | false alarms/hour 5.143

PASM v0.4 patient-wise ML holdout:
pasm_physionet            episode F1 0.500 | precision 0.500 | recall 0.500 | false alarms/hour 12.000
pasm_ml_decoder           episode F1 0.222 | precision 0.500 | recall 0.333 | false alarms/hour 4.333
pasm_ml_decoder_fpaware   episode F1 0.200 | precision 0.810 | recall 0.333 | false alarms/hour 4.000
pasm_ml_decoder_guarded   episode F1 0.200 | precision 0.810 | recall 0.333 | false alarms/hour 4.000

AFDB mini preset:
episode F1 0.933 | precision 1.000 | recall 0.889 | false alarms/hour 0.000

MITDB mini preset:
episode F1 0.542 | precision 0.625 | recall 0.500 | false alarms/hour 9.000
```

Status jakosci:

- Stara sciezka 1D CNN/embedding clustering zostala usunieta z kodu. Nie przywracaj `beat2vec.py`, `clustering.py`, `episodes.py` ani baseline `legacy_rr`, chyba ze uzytkownik wyraznie poprosi o eksperyment historyczny.
- PASM v0.4 ma FP-aware learned decoder bez nowych zaleznosci: capped class weights, hard negatives, tuned guarded config i ectopy-specific guard.
- Learned decoder ma juz nizszy FP/h niz `pasm_physionet`, ale nadal przegrywa episode F1 przez niski recall.
- Syntetyczny benchmark jest mocny i powinien byc traktowany jako regresja, ktorej nie wolno przypadkiem popsuc.
- AFDB po postprocessingu evidence ma wysoka precyzje na smoke/mini; `afdb/04015` nadal ma jeden pomijany krotki truth fragment, ale FP/h spadlo do 0.000.
- MITDB ektopia ma teraz flood-control ze wsparciem morfologicznym, ktory mocno obniza FP/h, ale `mitdb/203` i `mitdb/201` nadal sa glownymi stress case.
- To nadal nie jest pelna patient-wise walidacja kliniczna.

## Komendy Kontrolne

Uzywaj Pythona 3.11. Systemowy `python` moze wskazywac na Python 3.14 z problematycznym NumPy.

```bash
py -3.11 -m unittest discover -s tests -v
```

Syntetyczny train/holdout:

```bash
py -3.11 pasm_validation.py --train-records 30 --holdout-records 30 --seed 2026 --out PASM_VALIDATION.md
```

Patient-wise ML decoder:

```bash
py -3.11 pasm_ml_validation.py --preset tiny --out PASM_ML_VALIDATION.md
py -3.11 pasm_ml_validation.py --preset mini --out PASM_ML_BENCHMARK.md
```

Real-data smoke i mini:

```bash
py -3.11 pasm_realdata.py --preset smoke --out PASM_REALDATA_SMOKE.md
py -3.11 pasm_realdata.py --preset mini --out PASM_REALDATA_SUMMARY.md
```

MITDB smoke:

```bash
py -3.11 pasm_physionet.py --db mitdb --records 200 --max-seconds 900 --out PASM_PHYSIONET_VALIDATION.md
```

AFDB smoke:

```bash
py -3.11 pasm_physionet.py --db afdb --records 04015 04126 --max-seconds 900 --out PASM_AFDB_VALIDATION.md
```

Po `compileall` usun `__pycache__` przed koncowym statusem.

## Najblizsze Priorytety

### 1. Rozszerzyc Real-Data Benchmark

Obecnie real-data mini obejmuje:

- MITDB: `200`, `201`, `203`, `205`, `208`, pierwsze 900 s.
- AFDB: `04015`, `04043`, `04048`, `04126`, `04746`, pierwsze 1200 s.

Nastepny krok:

- Uzyc `pasm_ml_validation.py` jako pierwszego malego patient-wise splitu i rozszerzac go o kolejne rekordy.
- Dodac wiecej rekordow MITDB z ektopia i AFDB z AF, ale raportowac osobno rekordy bez truth.
- Utrzymac tabele inventory, per-record episode metrics, per-type F1 i false alarms/hour.

### 2. Zmniejszyc False Alarms Na AFDB

Aktualny problem:

- `AFDB 04015` ma teraz znacznie mniej falszywych tachy/AF-like alarmow po scalaniu AF i odrzucaniu bardzo krotkich tachy.
- `AFDB 04126` jest bardzo dobry.
- Nalezy sprawdzic, czy parametry evidence postprocessingu generalizuja poza mini preset.

Plik i funkcje:

- `pasm_physionet.py`
- `detect_fast_irregular_af(...)`
- `merge_physionet_evidence(...)`

Hipotezy do sprawdzenia:

- Wymagac wiekszego minimalnego czasu epizodu AF, np. 20-30 s, ale ostroznie: obecny postprocessing nie filtruje krotkich AF przed scaleniem.
- Zwiekszyc `min_beats` lub `win_beats`.
- Dodac warunek stabilnosci evidence: nie tylko srednia HR/CV/RMSSD, ale procent beatow w oknie spelniajacych nieregularnosc.
- Scalanie bliskiego AF-evidence jest wdrozone w `postprocess_physionet_episodes(...)` (`PHYSIONET_AF_MERGE_GAP_S=45.0`).
- Bardzo krotkie `sinus_tachy` sa odrzucane w PhysioNet postprocessingu (`PHYSIONET_MIN_TACHY_DURATION_S=3.0`), a tachy blisko AF jest supresowane (`PHYSIONET_AF_TACHY_MARGIN_S=10.0`).
- Nadal warto sprawdzic usuwanie izolowanych przedwczesnych epizodow przed pierwsza adnotacja AF na szerszym patient-wise train/holdout.
- Zamiast twardych progow, stroic AF evidence na malej train liscie AFDB i oceniac na holdout liscie.

Nie poprawiaj AFDB przez globalne wzmacnianie `af_like` w `pasm_rhythm.py`, bo poprzednio popsulo to syntetyczny benchmark. Lepiej rozwijac real-data evidence layer albo dodac jawnie konfigurowalny decoder dla PhysioNet.

### 3. Poprawic MITDB Ectopy Bez Zalewu Falszywych Alarmow

Aktualna poprawka:

- `detect_short_coupled_ectopy(...)` lapie krotkie runy short-coupled ectopy.
- Detektor uzywa teraz progu `short_rr_s=0.50`, `relative_rr_fraction=0.75` i scalenia epizodow do `merge_gap_s=1.0`.
- Dodany jest flood-control dla rekordow z nadmiarem kandydatow: `ECTOPY_FLOOD_RATE_PER_HOUR=30.0`, `ECTOPY_FLOOD_MIN_CONFIDENCE=0.40`, `ECTOPY_FLOOD_DENSITY_WINDOW_S=10.0`, `ECTOPY_FLOOD_MIN_DENSITY=6`, `ECTOPY_FLOOD_MIN_CANDIDATES=10`.
- MITDB 200 smoke: F1 0.667, precision 1.000, recall 0.500, FP/h 0.000.
- MITDB-mini: F1 0.542, precision 0.625, recall 0.500, FP/h 9.000.
- `reports/mitdb_203_diagnostic.html` pokazuje aktualnie `mitdb/203`: TP 6, FP 6, FN 6.
- Flood-control wymaga teraz wsparcia morfologicznego w zalanych fragmentach:
  - `ECTOPY_FLOOD_STRONG_MORPH_Z=0.55`;
  - `ECTOPY_FLOOD_DENSE_MORPH_Z=0.60`.

Nastepny krok:

- Po redukcji FP na `mitdb/203`, odzyskac recall bez powrotu do zalewu FP.
- Sprawdzic wiecej rekordow MITDB z `V`, `A`, `F`, ale z patient-wise holdout.
- Rozwazyc osobny beat-level PVC/SVE scorer:
  - RR pattern: premature beat + compensatory pause;
  - run pattern: kilka krotkich RR w serii;
  - morphology deviation, ale ostroznie, bo L2 na beat window bywa slabe na realnych danych;
  - local context: czy ektopia jest izolowana, bigeminy/trigeminy, czy run.

### 4. Uporzadkowac Raporty

Aktualne raporty sa uzyteczne, ale nie sa jeszcze ladnym benchmark suite.

Do zrobienia:

- Zapisywac parametry evidence layer w raportach.
- Raportowac liczbe rekordow, czas laczny, liczbe epizodow truth i pred.
- W raporcie odroznic:
  - syntetyczna walidacje regresyjna,
  - PhysioNet smoke tests,
  - przyszla patient-wise validation.

### 5. Nastepny Modelowy Krok

Aktualny pierwszy krok ML jest wdrozony:

- `pasm_dataset.py` buduje dataset tablicowy z cech PASM.
- `pasm_ml_decoder.py` trenuje softmax regression w NumPy.
- `pasm_ml_validation.py` robi patient-wise benchmark suite z presetami i analiza FP.

Nastepny modelowy krok:

- Nie zaczynac od duzego deep learningu bez szerszego patient-wise benchmarku.
- Rozszerzyc split o wiecej rekordow MITDB/AFDB.
- Odzyskac recall learned decodera bez powrotu do wysokiego FP/h.
- Porownac obecny softmax decoder z TCN/GRU/Transformer-lite dopiero po ustabilizowaniu datasetu.

## Rzeczy, Ktorych Nie Robic Pochopnie

- Nie oznaczac celu glownego jako kompletnego na podstawie obecnych smoke testow.
- Nie stroic globalnie `pasm_rhythm.py` pod jeden rekord AFDB, jesli syntetyczny holdout spada.
- Nie traktowac rekordow bez ground-truth epizodow jako sukcesu F1 1.0. `evaluate_physionet_records(..., skip_empty_truth=True)` juz domyslnie je pomija.
- Nie uzywac systemowego `python`, jesli odpala Python 3.14 z uszkodzonym NumPy. Uzywac `py -3.11`.
- Nie commitowac `__pycache__`.

## Minimalna Definicja Bezpiecznego Postepu

Po kazdej zmianie uruchom:

```bash
py -3.11 -m unittest discover -s tests -v
py -3.11 pasm_validation.py --train-records 30 --holdout-records 30 --seed 2026 --out PASM_VALIDATION.md
py -3.11 pasm_ml_validation.py --preset tiny --out PASM_ML_VALIDATION.md
py -3.11 pasm_ml_validation.py --preset mini --out PASM_ML_BENCHMARK.md
py -3.11 pasm_realdata.py --preset smoke --out PASM_REALDATA_SMOKE.md
py -3.11 pasm_realdata.py --preset mini --out PASM_REALDATA_SUMMARY.md
py -3.11 pasm_physionet.py --db mitdb --records 200 --max-seconds 900 --out PASM_PHYSIONET_VALIDATION.md
py -3.11 pasm_physionet.py --db afdb --records 04015 04126 --max-seconds 900 --out PASM_AFDB_VALIDATION.md
```

Akceptowalna zmiana powinna:

- zachowac zielone testy;
- nie obnizyc istotnie syntetycznego `pasm_tuned episode F1` ponizej okolo 0.80 bez bardzo dobrego powodu;
- nie pogorszyc jednoczesnie smoke i mini testow bez jasnego powodu;
- jasno zaktualizowac raporty `.md`.

## Obecny Kierunek Badawczy

PASM-Rhythm powinien isc w strone hybrydy:

```text
patient memory
+ rhythm-state decoder
+ AF fast-irregular evidence
+ beat-level ectopy evidence
+ SQI/uncertainty
+ patient-wise validation
```

Najwieksza luka do domkniecia: realna, patient-wise walidacja na wielu rekordach MITDB/AFDB/PhysioNet oraz redukcja false alarms/hour.
