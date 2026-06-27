# PASM-Rhythm: Patient-Adaptive State-Memory Rhythm Graph

## Status Dokumentu

Ten plik opisuje aktualny kierunek algorytmu po przejsciu repozytorium na
PASM-only. Stara sciezka embeddingow pojedynczych uderzen i klastrowania nie jest
juz czescia systemu decyzyjnego. PASM-Rhythm jest teraz glownym modelem
wykrywania arytmii, a obecna implementacja jest deterministycznym prototypem
badawczym, nie gotowym systemem klinicznym.

Aktualne pliki implementacyjne:

- `pasm_rhythm.py` - rdzen algorytmu PASM: cechy rytmu, patient memory, scoring
  stanow, graf evidence i dekoder epizodow.
- `pasm_dataset.py` - per-beat dataset cech PASM do uczenia i walidacji.
- `pasm_ml_decoder.py` - lekki uczony softmax decoder w NumPy.
- `pasm_ml_validation.py` - patient-wise walidacja deterministic PASM vs learned
  decoder.
- `pasm_validation.py` - syntetyczna walidacja train/holdout PASM-only.
- `pasm_physionet.py` - warstwa walidacji i dodatkowego evidence dla MITDB/AFDB.
- `pasm_realdata.py` - laczone presety real-data i raporty Markdown/HTML.

## Cel

PASM-Rhythm ma wykrywac epizody arytmii w zapisach EKG jako zmiany stanu rytmu w
czasie, a nie jako izolowane nietypowe uderzenia. Kluczowa idea jest taka, ze
"normalnosc" powinna byc liczona wzgledem danego pacjenta, aktualnego kontekstu
RR, morfologii uderzen oraz jakosci sygnalu.

Robocza teza:

> Arytmia nie jest tylko nietypowym uderzeniem. Arytmia jest odchyleniem od
> osobistego, dynamicznego stanu rytmu pacjenta, widocznym jako wspolna zmiana
> interwalow RR, lokalnej nieregularnosci, morfologii, przejsc stanow i
> wiarygodnosci sygnalu.

## Obecna Wersja Algorytmu

Aktualny PASM-Rhythm sklada sie z pieciu krokow:

```text
R-peaki + RR + beat windows + SQI
        |
        v
per-beat rhythm features
        |
        v
patient memory z fragmentu wysokiej jakosci
        |
        v
state scoring: normal / tachy / brady / AF-like / ectopy / noise
        |
        v
typed rhythm graph + duration-aware episode decoder
```

Wynikiem nie jest pojedyncza etykieta dla rekordu, tylko tabela epizodow:

```text
start_s, end_s, type, confidence, beats, mean_sqi, reason
```

Obecne typy stanow:

- `normal`
- `sinus_tachy`
- `sinus_brady`
- `af_like`
- `ectopic_like`
- `noise_uncertain`

## PASM v0.2: Learned State Scorer

PASM v0.2 dodaje pierwszy rzeczywisty komponent ML bez powrotu do ciezkiej
czarnej skrzynki. Deterministyczny PASM nadal jest baseline'em, ale jego cechy,
patient-relative z-score, morphology z-score oraz wlasne state scores sa
zapisywane jako tabela per-beat. Na tej tabeli trenowany jest lekki softmax
decoder w NumPy.

Przeplyw v0.2:

```text
deterministic PASM pipeline
        |
        v
per-beat feature dataset
        |
        v
softmax regression state scorer
        |
        v
istniejacy duration-aware episode decoder
        |
        v
patient-wise train/holdout report
```

To jest celowo maly krok ML:

- brak nowych zaleznosci typu PyTorch albo scikit-learn,
- normalizacja cech jest liczona tylko na train,
- klasy sa wazone, aby rzadkie epizody nie zniknely pod przewaga `normal`,
- learned decoder jest porownywany z `pasm_physionet`, a nie traktowany jako
  automatycznie lepszy.

## PASM v0.3: Benchmark And Error Analysis

PASM v0.3 nie wprowadza ciezszego modelu. Rozszerza walidacje learned decodera
tak, aby najpierw zrozumiec false positives:

- `tiny`, `mini`, `mitdb-mini`, `afdb-mini` jako jawne presety patient-wise,
- porownanie `pasm_physionet`, `pasm_ml_decoder` i `pasm_ml_decoder_guarded`,
- raport FP/h per rekord,
- rozbicie false positives wedlug typu stanu,
- top false-positive episodes z confidence, duration, SQI i przyczyna bledu.

Wariant `pasm_ml_decoder_guarded` dodaje konserwatywna warstwe po softmaxie:

- lekki bias na `normal`,
- minimalna srednia pewnosc epizodu,
- minimalne SQI epizodu,
- minimalna liczbe beatow per stan.

Cel v0.3: nie udawac, ze ML juz wygral, tylko zrobic benchmark, ktory pokazuje,
czy problem jest w beat-level scoringu, czy w dekoderze epizodow.

## PASM v0.4: FP-Aware Learned Decoder

PASM v0.4 zmniejsza false alarms learned decodera bez dodawania nowych
zaleznosci ML. Zmiany:

- capped class weights zamiast agresywnego fully-balanced weighting,
- hard-negative training: normalne beaty z train false-positive episodes dostaja
  wieksza wage,
- `pasm_ml_decoder_fpaware` jako osobny raportowany wariant,
- tuning guarded config na train,
- ectopy-specific guard wymagajacy wsparcia RR, morfologii albo
  deterministic `score_ectopic_like`.

Efekt: learned decoder zaczyna przegrywac z deterministic baseline'em mniej
przez FP/h, a bardziej przez niski recall. To jest lepszy problem do dalszego
badania.

## Wejscia

Rdzen PASM nie zaklada konkretnego detektora R-peakow. Dostaje juz przygotowane
dane rytmu:

- `r_times` - czasy R-peakow w sekundach,
- `rr_prev` - poprzedni interwal RR,
- `rr_next` - nastepny interwal RR,
- `beats` - opcjonalne okna morfologii wokol R-peaka,
- `sqi_at_r` - jakosc sygnalu przy uderzeniu,
- `rpeak_uncertainty` - niepewnosc detekcji R-peaka.

W docelowej wersji detektor R-peakow powinien zwracac nie tylko pozycje pikow,
ale tez wiarygodnosc kazdego wykrycia. Obecna implementacja jest gotowa na taka
informacje przez `rpeak_uncertainty`.

## Cechy Rytmu

Funkcja `compute_rhythm_features(...)` buduje lokalny opis kazdego uderzenia.
Najwazniejsze cechy:

- `hr` - chwilowa czestosc rytmu,
- `delta_rr` - zmiana RR wzgledem poprzedniego uderzenia,
- `rr_ratio` - relacja `rr_next / rr_prev`,
- `local_rr_median` - lokalna mediana RR,
- `local_rmssd` - lokalna nieregularnosc rytmu,
- `local_cv` - lokalny wspolczynnik zmiennosci RR,
- `sqi` - jakosc sygnalu,
- `rpeak_uncertainty` - niepewnosc R-peaka,
- `reliability` - wspolna wiarygodnosc: `SQI * (1 - uncertainty)`.

To jest wazna zmiana wzgledem klasycznego podejscia beat-level: uderzenie jest
oceniane w kontekscie sekwencji, a nie jako samotny waveform.

## Patient Memory

Funkcja `build_patient_memory(...)` tworzy osobista norme pacjenta z pierwszego
fragmentu zapisu, preferujac uderzenia o wysokiej jakosci:

```text
patient_memory = {
  morphology_prototype,
  morphology_scale,
  rr_median,
  rr_mad,
  rmssd_median,
  rmssd_mad,
  sqi_median,
  n_baseline_beats
}
```

Pamiec pacjenta jest celowo konserwatywna. Jezeli we wczesnym fragmencie jest
wystarczajaco duzo dobrych uderzen, tylko one buduja baseline. Jezeli jakosc jest
slabsza, algorytm schodzi do fallbacku opartego na dostepnych poprawnych RR.

Najwazniejsza konsekwencja:

```text
to samo RR albo ta sama morfologia moga miec inne znaczenie u roznych pacjentow
```

Dlatego PASM uzywa robust z-score wzgledem pacjenta, a nie tylko globalnych
progowych regul.

## State Scoring

Funkcja `score_pasm_states(...)` przelicza cechy na evidence score dla stanow
rytmu. To nie sa jeszcze klinicznie skalibrowane prawdopodobienstwa, tylko
deterministyczne, porownywalne wskazniki evidence, normalizowane softmaxem.

Glowna logika:

- `sinus_tachy` rosnie, gdy HR przekracza prog szybkiego rytmu i sygnal jest
  wiarygodny.
- `sinus_brady` rosnie, gdy HR spada ponizej progu wolnego rytmu i sygnal jest
  wiarygodny.
- `af_like` rosnie przy lokalnej nieregularnosci RR, zmianach `rr_ratio` i
  podwyzszonym `local_rmssd`.
- `ectopic_like` rosnie przy skokach RR oraz, jezeli sa dostepne beat windows,
  odchyleniu morfologii od prototypu pacjenta.
- `noise_uncertain` rosnie przy niskim SQI lub wysokiej niepewnosci R-peaka.
- `normal` spada, gdy rosna odchylenia RR, nieregularnosc, odchylenie morfologii
  lub evidence szumu.

Wazny mechanizm:

```text
arrhythmia evidence jest bramkowane przez reliability
noise evidence rosnie, gdy reliability spada
```

To oznacza, ze zly sygnal nie powinien automatycznie udawac arytmii. Powinien
czesciej trafic do `noise_uncertain`.

## Rhythm Graph

Funkcja `build_rhythm_graph(...)` tworzy zwarta reprezentacje grafowa:

- wierzcholki `beat_i` zawieraja cechy rytmu i najlepszy stan,
- wierzcholki `state_*` reprezentuja mozliwe stany rytmu,
- krawedzie `temporal_next` lacza kolejne uderzenia,
- krawedzie `state_likelihood` lacza beat ze stanami o wystarczajacym evidence.

Obecnie graf jest lekka, interpretowalna struktura danych. Nie jest jeszcze GNN.
Jego rola na tym etapie:

- ulatwia debugowanie decyzji,
- zachowuje powiazanie beat -> evidence -> stan,
- przygotowuje architekture pod przyszly relation-aware decoder albo GNN.

## Episode Decoder

Funkcja `decode_pasm_episodes(...)` zamienia beat-level state scores na epizody.
Dekoder:

- wybiera najlepszy stan dla kazdego uderzenia,
- odrzuca `normal`,
- stosuje minimalny prog pewnosci per stan,
- scala krotkie przerwy do `merge_gap_beats`,
- wymaga minimalnej dlugosci epizodu per stan,
- zapisuje srednia pewnosc, liczbe beatow, SQI i tekstowy powod decyzji.

Domyslne minimalne dlugosci:

```text
sinus_tachy      5 beatow
sinus_brady      5 beatow
af_like          8 beatow
ectopic_like     3 beaty
noise_uncertain  2 beaty
```

W walidacji syntetycznej progi dekodera sa strojone na train i oceniane na
holdout. Aktualny tuned zestaw progow:

```text
af_like          0.400
ectopic_like     0.280
noise_uncertain  0.300
sinus_brady      0.240
sinus_tachy      0.240
```

## Warstwa PhysioNet Evidence

Dla danych MITDB/AFDB istnieje dodatkowa warstwa w `pasm_physionet.py`. Nie
zastepuje ona PASM, tylko dodaje evidence specyficzne dla sposobu adnotowania
tych baz.

AFDB:

- dodatkowe wykrywanie fast-irregular AF,
- scalanie pobliskich fragmentow AF,
- supresja bardzo krotkich tachykardii przylegajacych do AF.

MITDB:

- wykrywanie short-coupled ectopy,
- filtrowanie zalewu kandydatow ektopowych,
- wymaganie wsparcia morfologicznego w trudnych fragmentach.

Ta warstwa jest praktyczna dla benchmarkow, ale docelowo powinna zostac
zastapiona bardziej ogolnym, uczonym albo semi-Markov evidence decoderem.

## Aktualne Wyniki Walidacyjne

Syntetyczny train/holdout PASM-only:

```text
pasm_tuned    episode F1 0.834 | precision 0.773 | recall 0.917 | FP/h 17.581
pasm_default  episode F1 0.749 | precision 0.733 | recall 0.792 | FP/h 19.256
```

Real-data smoke preset:

```text
pasm_physionet episode F1 0.822 | precision 1.000 | recall 0.722 | FP/h 0.000
```

Real-data mini preset:

```text
pasm_physionet episode F1 0.710 | precision 0.786 | recall 0.667 | FP/h 5.143
```

PASM v0.4 patient-wise ML holdout:

```text
pasm_physionet            episode F1 0.500 | precision 0.500 | recall 0.500 | FP/h 12.000
pasm_ml_decoder           episode F1 0.222 | precision 0.500 | recall 0.333 | FP/h 4.333
pasm_ml_decoder_fpaware   episode F1 0.200 | precision 0.810 | recall 0.333 | FP/h 4.000
pasm_ml_decoder_guarded   episode F1 0.200 | precision 0.810 | recall 0.333 | FP/h 4.000
```

Interpretacja:

- syntetyka jest dobra jako regresja kodu i zachowania epizodowego,
- AFDB wyglada obiecujaco po evidence postprocessingu,
- MITDB ektopia, szczegolnie `mitdb/203` i `mitdb/201`, pozostaje glownym stress
  case,
- learned decoder ma juz nizszy FP/h niz deterministic baseline, ale nadal
  przegrywa episode F1 przez niski recall,
- to nadal nie jest pelna patient-wise walidacja kliniczna.

## Co Jest Oryginalnym Kierunkiem

Najsilniejszy wklad badawczy PASM:

> Pacjentozalezny graf stanow rytmu, ktory laczy osobista pamiec normalnosci,
> relacje RR-morfologia-SQI, sekwencyjne przejscia stanow i jawna niepewnosc
> epizodow.

Elementy skladowe, ktore same w sobie moga juz istniec:

- lokalne cechy HRV,
- modele morfologii ECG,
- Transformer/TCN/Mamba dla ECG,
- grafowe modele EKG,
- patient-specific adaptation,
- uncertainty-aware medical time series.

Nowosc powinna lezec w kompozycji:

- patient memory jako aktywny punkt odniesienia,
- typed rhythm graph beat-window-state,
- duration-aware episode decoder,
- SQI jako zrodlo niepewnosci, a nie tylko filtr,
- patient-wise adaptacja bez pelnych etykiet dla pacjenta,
- raportowalna decyzja: stan, pewnosc, SQI, evidence i powod.

## Docelowa Architektura Badawcza

Obecny system ma deterministyczny baseline i pierwszy lekki learned scorer.
Docelowo mozna go rozwinac bez zmiany filozofii PASM:

```text
ECG signal
  -> R-peak detector with uncertainty
  -> beat/window representation
  -> patient memory
  -> sequence model / relation-aware graph model
  -> duration-aware state decoder
  -> calibrated episode report
```

Mozliwe moduly uczone:

- lekki ResNet1D lub TCN dla morfologii,
- Transformer/Mamba/S4 dla dluzszego kontekstu rytmu,
- relation-aware attention zamiast recznego grafu,
- neural CRF / semi-Markov decoder dla onset-offset,
- kalibracja niepewnosci na poziomie epizodu.

Wazne: jezeli dodajemy uczenie, nie powinno ono wracac do starego schematu
"embedding + clustering jako decyzja". Uczony modul powinien wzmacniac PASM:
pamiec pacjenta, stan rytmu, niepewnosc i epizody.

## Plan Rozwoju

### Etap 1: Ustabilizowac PASM Deterministyczny

- Utrzymac zielone testy jednostkowe.
- Rozszerzyc raporty o parametry dekodera i evidence layer.
- Poprawic MITDB ektopie bez wzrostu false alarms/hour.
- Zweryfikowac AFDB evidence na szerszym patient-wise holdout.

### Etap 2: Zbudowac Dataset Tablicowy

Status: wdrozone jako `pasm_dataset.py`.

Dataset per-beat zawiera:

- HR, RR, `delta_rr`, `rr_ratio`,
- `local_cv`, `local_rmssd`, lokalna mediana RR,
- patient-relative RR z-score,
- morphology distance/prototype features,
- SQI i rpeak uncertainty,
- etykiety beat-level i episode-level z MITDB/AFDB.

To pozwala porownac deterministyczny PASM z lekkim learned decoderem.

### Etap 3: Uczony Decoder PASM-Compatible

Status: pierwsza wersja wdrozona jako `pasm_ml_decoder.py`.

Obecny model:

- softmax regression w NumPy,
- median fill + standaryzacja z train,
- balanced class weights,
- wyjscie kompatybilne z `PASM_STATES`,
- dekodowanie epizodow przez istniejacy `decode_pasm_episodes(...)`.

Nastepne modele powinny byc nadal male i interpretowalne:

- TCN/GRU/Transformer-lite na cechach rytmu,
- osobny head dla stanu i niepewnosci,
- duration-aware postprocessing,
- patient-wise train/holdout.

### Etap 4: Pelny Rhythm Graph

Rozwinac obecny graf evidence do grafu beat-window-state:

- beat nodes,
- rhythm window nodes,
- patient prototype nodes,
- state nodes,
- typed edges dla czasu, podobienstwa, evidence i niepewnosci.

Potem porownac:

- deterministyczny PASM,
- TCN/Transformer na cechach,
- relation-aware attention,
- GNN.

## Metryki Sukcesu

PASM-Rhythm powinien byc oceniany epizodowo, nie tylko beat-level:

- episode-level F1,
- precision/recall epizodow,
- false alarms per hour,
- onset/offset error,
- typed F1 per arytmia,
- calibration error confidence,
- performance przy niskim SQI,
- generalizacja patient-wise.

Robocza definicja sukcesu:

- mniej falszywych alarmow na godzine,
- lepsze granice poczatku i konca epizodu,
- mniej mylenia artefaktu z arytmia,
- stabilna generalizacja na pacjentach niewidzianych w train,
- raportowalna niepewnosc zamiast wymuszonej decyzji.

## Minimalny Pseudokod Aktualnej Implementacji

```python
from pasm_rhythm import run_pasm_rhythm

result = run_pasm_rhythm(
    r_times,
    rr_prev,
    rr_next=rr_next,
    beats=beats,
    sqi_at_r=sqi_at_r,
    rpeak_uncertainty=rpeak_uncertainty,
    win_beats=10,
    memory_warmup_beats=300,
)

features = result["features"]
patient_memory = result["patient_memory"]
state_scores = result["state_scores"]
rhythm_graph = result["graph"]
episodes = result["episodes"]
```

## Powiazane Kierunki Literatury Do Sprawdzenia

Te obszary trzeba sledzic, zeby nowosc PASM nie polegala tylko na nazwie:

- ECG foundation models,
- self-supervised ECG,
- contrastive ECG,
- Transformer/Mamba/TCN dla sygnalow EKG,
- grafowe modele ECG,
- patient-specific ECG adaptation,
- uncertainty-aware medical time series,
- semi-Markov i CRF dla segmentacji epizodow.

Przykladowe punkty odniesienia:

- ECG-FM: https://arxiv.org/html/2408.05178v1
- ECG foundation model review: https://arxiv.org/html/2410.19877v3
- Poly-window contrastive ECG: https://arxiv.org/html/2508.15225v1
- ECG-GraphNet: https://www.heartrhythmopen.com/article/S2666-5018%2825%2900162-X/fulltext

## Haslo Projektu

```text
PASM-Rhythm = Patient-Adaptive State-Memory Rhythm Graph
```

Sedno:

```text
personalna norma pacjenta
+ lokalny kontekst RR
+ morfologia wzgledem prototypu pacjenta
+ SQI jako niepewnosc
+ graf evidence beat-state
+ dekoder epizodow
```
