# Tunel-aero ✈️🚁

**Wirtualne środowisko testowe dla dronów i samolotów bezzałogowych w realistycznych warunkach —
także dla Twoich własnych konstrukcji zaprojektowanych w CAD.**

[![Otwórz aplikację w przeglądarce](https://img.shields.io/badge/aplikacja-otwórz_w_przeglądarce-1f6fb2)](https://ddbd-113.github.io/Tunel-aero/)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/ddbd-113/Tunel-aero)

- **W przeglądarce (bez instalacji):** <https://ddbd-113.github.io/Tunel-aero/> — wgrywasz pliki STL/OBJ
  swojego samolotu, dostajesz analizę stateczności i wyważenia, wybierasz warunki i awarie, oglądasz lot w 3D.
  Python działa w przeglądarce (Pyodide), Twoje pliki nie są nigdzie wysyłane.
- **W GitHub Codespaces:** przycisk powyżej → pełne środowisko w chmurze z terminalem (`tunel-aero …`),
  aplikacja otwiera się automatycznie na porcie 8000.
- **Lokalnie:** `pip install -e .` (instrukcja niżej).

Tunel-aero symuluje lot (6 stopni swobody) wielowirnikowców i samolotów w warunkach, które
spotyka się w prawdziwym świecie: porywisty wiatr nad miastem, turbulencja, mikroburst z burzy,
kominy termiczne, mróz i zimny akumulator, upał na dużej wysokości, deszcz, oblodzenie w chmurze,
utrata/zagłuszanie/spoofing GPS, zakłócenia magnetometru, awarie silników i sterów, utrata łącza.

Każdy test to plik YAML z warunkami, misją, awariami i **kryteriami zaliczenia (PASS/FAIL)**.
Wynikiem jest raport HTML z wykresami, log CSV i **odtwarzacz 3D lotu** w przeglądarce.

```text
tunel-aero run scenarios/multirotor/07_burza_mikroburst.yaml

  Quad X500 - burza: ulewa, mikroburst, silna turbulencja
  wynik: PASS  (wylądowano)
    [OK] brak katastrofy: tak  (próg: True)
    [OK] min. wysokość w misji [m]: 29.72  (próg: 10)
    [OK] maks. pochylenie [deg]: 34.95  (próg: 40)
    [OK] błąd miejsca lądowania [m]: 1.95  (próg: 8)
  raport:     wyniki/07_burza_mikroburst/raport.html
  odtwarzacz: wyniki/07_burza_mikroburst/replay.html
```

---

## Spis treści
1. [Instalacja](#instalacja)
2. [Szybki start](#szybki-start)
3. [Własny samolot z CAD](#własny-samolot-z-cad)
4. [Aplikacja w przeglądarce, GitHub Pages i Codespaces](#aplikacja-w-przeglądarce-github-pages-i-codespaces)
5. [Co jest modelowane](#co-jest-modelowane)
6. [Gotowe scenariusze](#gotowe-scenariusze)
7. [Własny scenariusz (YAML)](#własny-scenariusz-yaml)
8. [Własny pojazd (parametry)](#własny-pojazd)
9. [Monte Carlo](#monte-carlo)
10. [Wirtualny tunel aerodynamiczny](#wirtualny-tunel-aerodynamiczny)
11. [API Pythona i własne regulatory (styl Gymnasium)](#api-pythona-i-własne-regulatory)
12. [Architektura](#architektura)
13. [Ograniczenia i wiarygodność](#ograniczenia-i-wiarygodność)

---

## Instalacja

Wymagany Python ≥ 3.10.

```bash
git clone https://github.com/ddbd-113/Tunel-aero.git
cd Tunel-aero
pip install -e ".[dev]"      # numpy, pyyaml, matplotlib (+ pytest)
pytest -q                    # 54 testy, ~20 s
```

Odtwarzacz 3D ładuje bibliotekę three.js z `cdn.jsdelivr.net`, więc przy oglądaniu potrzebny jest internet.

## Szybki start

```bash
tunel-aero list                                              # lista scenariuszy
tunel-aero run scenarios/multirotor/02_wiatr_porywisty_miasto.yaml
tunel-aero run scenarios/fixed_wing/02_mikroburst_nisko.yaml --seed 5

# nadpisanie dowolnego parametru bez edycji pliku:
tunel-aero run scenarios/multirotor/01_spokojny_lot.yaml \
    --set environment.wind.speed=11 --set environment.temperature_c=-10

tunel-aero suite --jobs 4                                    # wszystkie scenariusze (test regresyjny)
tunel-aero montecarlo scenarios/montecarlo/quad_wiatr_miasto_mc.yaml --runs 50 --jobs 4
tunel-aero tunnel vehicles/fixed_wing_3kg.yaml               # charakterystyki aerodynamiczne i osiągi
```

Wyniki trafiają do `wyniki/<nazwa>/`:

| plik | zawartość |
|---|---|
| `raport.html` | werdykt PASS/FAIL, kryteria, kluczowe wielkości, lista zdarzeń, wykresy |
| `replay.html` | odtwarzacz 3D: tor lotu, model pojazdu, wektor wiatru, mikrobursty/termika, HUD, oś czasu ze zdarzeniami |
| `log.csv` | pełny log (25 Hz): stan, wiatr, atmosfera, bateria, sterowanie, estymata nawigacji |
| `przeglad.png`, `srodowisko.png` | wykresy |

Kod wyjścia `run`/`suite` ≠ 0, gdy test nie spełnił oczekiwań — można go wpiąć w CI
(przykład: `.github/workflows/tests.yml`).

## Własny samolot z CAD

Projektujesz samolot w CAD (Fusion 360, SolidWorks, Onshape, FreeCAD, Blender…)? Wyeksportuj
**każdą część jako osobny plik STL** (albo jeden OBJ z nazwanymi obiektami) i opisz ją krótkim plikiem
`aircraft.yaml`. Tunel-aero sam:

1. **przetnie siatki płaszczyznami** i wyznaczy przekroje skrzydła i usterzenia: cięciwy, rozpiętość,
   powierzchnię, średnią cięciwę aerodynamiczną, wznios, skos, zaklinowanie, grubość profilu
   (obsługuje też usterzenie motylkowe V i latające skrzydła z elewonami),
2. policzy **masę, środek ciężkości i momenty bezwładności** z siatek (bryła lub powłoka) i elementów
   skupionych (silnik, akumulator, serwa…) — albo przyjmie wartości z Twojego CAD,
3. wyznaczy **pochodne stateczności metodą siatki wirowej (VLM, jak w AVL/XFLR5)** z wychyleniami
   sterów, doda wpływ kadłuba (DATCOM/Raymer) i **opór z rzeczywistej powierzchni zwilżonej siatek**,
4. pokaże **punkt neutralny, zapas stateczności i zalecany zakres środka ciężkości** (w Twoich
   współrzędnych CAD), prędkość przeciągnięcia, trym, moc w przelocie i ostrzeżenia projektowe,
5. zbuduje model do symulatora — samolot lata potem wszystkimi scenariuszami z autopilotem,
   a w odtwarzaczu 3D widać **Twoją siatkę z CAD**.

```bash
tunel-aero cad examples/moj_samolot/aircraft.yaml          # raport projektu (wyważenie, stateczność, osiągi)
tunel-aero run scenarios/fixed_wing/02_mikroburst_nisko.yaml --vehicle examples/moj_samolot/aircraft.yaml
tunel-aero suite scenarios/fixed_wing --vehicle examples/moj_samolot/aircraft.yaml   # wszystkie testy samolotowe
```

Przykład (`examples/moj_samolot/` — trener 1,4 m, pliki STL wygenerowane skryptem `generuj_stl.py`):

```yaml
type: cad_aircraft
name: "Trener 1.4 m"
units: mm                          # jednostki plików: mm | cm | m | in
axes: {forward: -x, up: +z}        # gdzie w CAD jest nos i "góra" samolotu
parts:                             # role: wing | htail | vtail | fuselage | other
  skrzydlo:   {file: skrzydlo.stl, role: wing, mass: 0.40}
  usterzenie: {file: ster_wysokosci.stl, role: htail, mass: 0.035}
  statecznik: {file: statecznik.stl, role: vtail, mass: 0.025}
  kadlub:     {file: kadlub.stl, role: fuselage, mass: 0.28}
  # albo jeden plik OBJ:  skrzydlo: {file: "samolot.obj#Skrzydlo", role: wing}
mass:
  # total: 1.15                    # masa do lotu; bez mas części rozłoży się po ich powierzchni
  # cg: [318, 0, 40]               # środek ciężkości z wyważenia lub z CAD
  # inertia: [0.06, 0.08, 0.13, 0] # Ixx, Iyy, Izz, Ixz [kg m^2] z CAD (domyślnie liczone)
  components:                      # elementy skupione, pozycje w układzie CAD
    - {name: silnik,     mass: 0.110, position: [10, 0, 10]}
    - {name: akumulator, mass: 0.185, position: [60, 0, 5]}
surfaces:
  wing:
    airfoil: {cl_max: 1.3, zero_lift_alpha_deg: -2.1, cm0: -0.05}    # NACA 2412
    controls: [{type: aileron, span: [0.55, 0.95], chord_fraction: 0.25, max_deg: 20}]
  htail:
    controls: [{type: elevator, span: [0, 1], chord_fraction: 0.35, max_deg: 25}]
  vtail:
    controls: [{type: rudder, span: [0, 1], chord_fraction: 0.4, max_deg: 25}]
  # zamiast siatki można podać przekroje: sections: [[x_LE, y, z, cięciwa, skręcenie_deg], ...]
propulsion: {position: [-20, 0, 10], kv: 1000, diameter_in: 10, pitch_in: 5}
battery: {cells: 3, capacity_ah: 2.2}
cruise_speed: 13
```

**Eksport z CAD — wskazówki:**

| program | eksport | typowe osie (`axes`) |
|---|---|---|
| Fusion 360 | prawy klik na bryle → *Save As Mesh* → STL (binarny), jednostki mm | `up: +y` (lub `+z` przy „Z up”) |
| SolidWorks | *Zapisz jako* → STL, *Opcje* → „zapisz wszystkie części jako osobne pliki” | `up: +y` |
| Onshape | prawy klik na części → *Export* → STL | `up: +z` |
| FreeCAD | *Plik → Eksportuj* → STL/OBJ | `up: +z` |
| Blender | *File → Export → Wavefront OBJ* (obiekty zachowają nazwy) | `up: +z` |

Kierunek nosa (`forward`) zależy od tego, jak narysowałeś samolot. Po analizie sprawdź podgląd 3D —
strzałka „NOS” musi wskazywać przód. Ważne: części powinny być **zamkniętymi bryłami**; skrzydło może
być w całości lub jako jedna połówka (zostanie odbite). Lotki, ster wysokości i kierunku nie muszą być
osobnymi bryłami — ich zakres podajesz w `controls`.

## Aplikacja w przeglądarce, GitHub Pages i Codespaces

Aplikacja webowa (`tunel_aero/web/`) uruchamia **ten sam kod Pythona** w przeglądarce (Pyodide
w Web Workerze), więc działa jako statyczna strona — bez serwera i bez kosztów:

1. **Pojazd** — wgraj pliki STL/OBJ (role części rozpoznawane z nazw plików, np. `skrzydlo.stl`,
   `ster_wysokosci.stl`, `statecznik.stl`, `kadlub.stl`), wpisz masy i napęd, kliknij *Analizuj samolot*
   (albo wybierz gotowy pojazd),
2. **Warunki testu** — scenariusz bazowy + wiatr, teren, turbulencja, podmuchy, temperatura, wysokość
   terenu, deszcz, oblodzenie i awarie (GPS, łącze, Pitot, stery, silnik, ogniwo),
3. **Wyniki** — PASS/FAIL, kryteria, zdarzenia, wykresy i odtwarzacz 3D z Twoją siatką; pobierz
   `aircraft.yaml`, model dla symulatora, log CSV, scenariusz i odtwarzacz.

| gdzie | jak |
|---|---|
| GitHub Pages | workflow `.github/workflows/pages.yml` publikuje stronę przy każdym pushu do domyślnej gałęzi. **Jednorazowo:** *Settings → Pages → Build and deployment → Source: GitHub Actions*, potem *Actions → Strona (GitHub Pages) → Run workflow*. Adres: `https://<użytkownik>.github.io/Tunel-aero/` |
| GitHub Codespaces | *Code → Codespaces → Create codespace* (lub przycisk na górze). `.devcontainer/` instaluje pakiet i uruchamia aplikację na porcie 8000 |
| lokalnie | `tunel-aero web` → <http://localhost:8000> (`--build-only --out site` tylko buduje stronę) |

## Co jest modelowane

| obszar | model | źródło / uwagi |
|---|---|---|
| dynamika | bryła sztywna 6-DOF, kwaterniony, RK4 (200 Hz multirotor, 100 Hz samolot) | NED / FRD |
| atmosfera | ISA do 47 km, temperatura przy ziemi lub odchyłka od ISA, QNH, **tendencja ciśnienia** (front), wilgotność | ICAO/ISO 2533, równanie Bucka |
| wiatr średni | profil logarytmiczny zależny od **kategorii terenu** (woda, śnieg, teren otwarty, pola, przedmieścia, las, miasto), prawo potęgowe, **tabela warstw** (np. z sondażu) | EN 1991-1-4 (Eurokod 1) |
| turbulencja | Dryden (filtry kształtujące), poziomy light/moderate/severe lub `auto` z warstwy przyziemnej (σ zależne od wiatru i szorstkości) | MIL-F-8785C / MIL-HDBK-1797 |
| podmuchy | dyskretne 1-cos, losowe (proces Poissona) | MIL-F-8785C |
| mikroburst | analityczny model prądu zstępującego z wypływem przy ziemi, przemieszczanie komórki | Oseguera & Bowles, NASA TM-100632 |
| termika | kominy z noszeniem i pierścieniem opadania, dryf z wiatrem | model Gedeona |
| deszcz | spadek CL / wzrost CD / spadek ciągu, pęd kropel | półempiryczny (NASA, Bezos i in.) |
| oblodzenie | narastanie lodu w przechłodzonej chmurze (LWC, temperatura, prędkość), degradacja CL, CD, kąta krytycznego, ciągu śmigieł; **zamarzanie rurki Pitota** | Bragg i in. (`C_iced = (1+ηk)C`) |
| napęd | silnik BLDC (KV, spadek obrotów z napięciem), śmigło `C_T(J)`, moc z teorii strumieniowej, siła H, **efekt przypowierzchniowy**, **pierścień wirowy (VRS)** | Cheeseman-Bennett, Glauert |
| akumulator | krzywa OCV(SOC), rezystancja wewn., **pojemność i rezystancja zależne od temperatury**, nagrzewanie I²R / chłodzenie | typowe dane LiPo |
| samolot | pochodne stateczności + nieliniowe przeciągnięcie (opadanie skrzydła), serwa (opóźnienie, szybkość, limity), VNE, przeciążenie | Beard & McLain |
| czujniki | IMU (szum, bias, drgania), GPS (błąd Gaussa-Markowa, opóźnienie), barometr (**błąd wysokości przy niestandardowej temperaturze**), magnetometr (zakłócenia od prądu), Pitot (IAS) | |
| nawigacja | model błędów EKF: dryf GPS, **nawigacja zliczeniowa po utracie GPS**, skok po odzyskaniu, spoofing | autopilot steruje na podstawie estymaty, nie prawdy |
| autopiloty | multirotor: kaskada pozycja→prędkość→ciąg→orientacja→prędkości kątowe→mikser z priorytetami; samolot: TECS, prowadzenie po trasie z antycypacją zakrętów, loiter | w stylu PX4/ArduPilot |
| failsafe | niski poziom baterii (RTL/lądowanie), utrata łącza, utrata GPS, wykrycie awarii silnika (realokacja ciągu), zablokowana lotka (sterowanie sterem kierunku) | |
| kontakt z ziemią | sprężysto-tłumiący z tarciem, detektor lądowania, kryteria rozbicia (prędkość pionowa/pozioma, wywrócenie) | |

### Awarie (`failures:` w scenariuszu)

`motor`, `propeller_damage`, `control_surface`, `gps_loss`, `gps_degraded`, `gps_spoofing`,
`mag_interference`, `pitot_blocked`, `battery_cell`, `link_loss`, `imu_vibration` — każda z czasem
`t` i opcjonalnym `duration`. Szczegóły parametrów: `tunel_aero/failures.py`.

## Gotowe scenariusze

| scenariusz | warunki | oczekiwany wynik |
|---|---|---|
| `multirotor/01_spokojny_lot` | ISA, bez wiatru — punkt odniesienia | PASS |
| `multirotor/02_wiatr_porywisty_miasto` | 9 m/s nad miastem, turbulencja miejska, podmuchy 3–8 m/s | PASS |
| `multirotor/03_awaria_silnika_heksa` | heksakopter traci silnik w locie, wykrycie po 0,4 s, lądowanie awaryjne | PASS |
| `multirotor/04_mroz_zimna_bateria` | −15 °C, niepodgrzany akumulator 60% → failsafe RTL | PASS |
| `multirotor/05_wysokogorze_upal` | 4000 m n.p.m., +18 °C, ładunek — praca na granicy ciągu | PASS (na granicy) |
| `multirotor/06_utrata_gps` | zagłuszanie GNSS przez 25 s w wietrze, nawigacja zliczeniowa | PASS |
| `multirotor/07_burza_mikroburst` | ulewa 40 mm/h, silna turbulencja, przesuwający się mikroburst, spadek ciśnienia | PASS |
| `multirotor/08_oblodzenie_marznaca_mgla` | marznąca mgła −4 °C — lód na śmigłach | **FAIL** (demonstracja) |
| `multirotor/09_mini_dron_silny_wiatr` | dron 249 g, wiatr na wysokości > prędkość maksymalna | **FAIL** (demonstracja) |
| `multirotor/10_spoofing_gps` | fałszywy sygnał przesuwa pozycję — autopilot „nie widzi” błędu | **FAIL** (demonstracja) |
| `fixed_wing/01_patrol` | patrol 4 odcinki, lekki wiatr | PASS |
| `fixed_wing/02_mikroburst_nisko` | przelot na 80 m przez mikroburst (wypływ 14 m/s) | PASS |
| `fixed_wing/03_oblodzenie_chmura` | lot w chmurze −7 °C, bez grzania Pitota → utrata wskazań prędkości → zderzenie | **FAIL** (demonstracja) |
| `fixed_wing/04_termika_turbulencja` | upał +31 °C, kominy termiczne, umiarkowana turbulencja | PASS |
| `fixed_wing/05_zablokowana_lotka` | lotka zacina się na +3°, rekonfiguracja na ster kierunku | PASS |
| `fixed_wing/06_utrata_lacza` | utrata łącza → powrót i krążenie nad startem | PASS |

Scenariusze „demonstracyjne” mają `expect_pass: false` — pokazują, **jak i kiedy** pojazd zawodzi.
`tunel-aero suite` traktuje je jako zgodne z oczekiwaniem, gdy kończą się porażką.

Scenariusze samolotowe domyślnie latają samolotem 3 kg, ale z opcją `--vehicle` testują dowolny samolot
(także z CAD); prędkość lotu bierze się wtedy z `cruise_speed` pojazdu, a kryterium `min_stall_margin`
(minimalna IAS / prędkość przeciągnięcia) jest niezależne od wielkości samolotu. Oczekiwane wyniki w tabeli
dotyczą pojazdu domyślnego — inny samolot może wypaść lepiej lub gorzej i właśnie po to są te testy.

## Własny scenariusz (YAML)

```yaml
name: "Mój test - inspekcja mostu przy silnym wietrze"
description: "Opis widoczny w raporcie"
seed: 42                    # ziarno losowe (turbulencja, szumy) - powtarzalność
duration: 180               # [s]
# extends: ../multirotor/02_wiatr_porywisty_miasto.yaml   # opcjonalnie: dziedzicz i nadpisuj
vehicle: ../../vehicles/quad_x500.yaml
vehicle_overrides:          # zmiany parametrów pojazdu (np. ładunek)
  mass: 2.4
  battery: {soc: 0.9, temperature_c: 25, insulated: true}   # podgrzany, izolowany akumulator

environment:
  ground_elevation: 250     # m n.p.m.
  temperature_c: 5          # przy ziemi (alternatywnie isa_offset)
  qnh_hpa: 1008
  humidity: 0.8
  pressure_tendency_hpa_h: -2        # front - dryf wysokości barometrycznej
  rain_mm_h: 5
  wind:
    speed: 10               # [m/s] jak w prognozie: 10 m nad terenem otwartym
    direction: 270          # SKĄD wieje [deg]
    terrain: water          # water|snow|open|farmland|suburbs|forest|city (lub roughness: z0)
    turbulence: auto        # auto|none|light|moderate|severe|<sigma m/s>
    # profile: layers
    # layers: [[10, 6, 250], [100, 12, 260], [300, 18, 270]]   # [wysokość, prędkość, kierunek]
    gusts: [{t: 40, duration: 3, speed: 8, direction: 280, vertical: -2}]
    random_gusts: {rate_per_min: 2, speed: [3, 7], duration: [1, 4]}
    microbursts: [{north: 300, east: 0, radius: 400, max_outflow: 12, t_start: 20}]
    thermals: [{north: 100, east: 50, radius: 80, core_updraft: 3}]
  icing: {lwc: 0.3, cloud_base: 200, cloud_top: 600, deicing: false}

sensors:                    # nadpisanie domyślnych parametrów czujników (sensors.py)
  gps: {sigma_h: 0.05, sigma_v: 0.1}  # np. RTK
navigation: realistic       # realistic | perfect

mission:                    # multirotor
  takeoff_altitude: 20
  cruise_speed: 6
  waypoints:                # [północ, wschód, wysokość AGL, (zawis s)]
    - [100, 0, 20]
    - [100, 60, 25, 10]
  land: true                # lub return_home: true
  low_battery_soc: 0.25
  low_battery_action: rtl   # rtl | land
  gps_loss_action: land     # continue | land
  link_loss_action: rtl

failures:
  - {type: motor, t: 60, motor: 2, efficiency: 0.0, detection_delay: 0.5}
  - {type: gps_loss, t: 90, duration: 15}

criteria:                   # wszystkie muszą być spełnione -> PASS
  no_crash: true
  mission_complete: true
  max_track_error: 5.0      # [m] liczona z RZECZYWISTEJ pozycji
  max_tilt_deg: 40
  min_battery_soc: 0.2
  max_landing_error: 3.0
```

Samolot: `mission.airspeed` (IAS), `start: {north, east, altitude, heading_deg}`, `loiter_radius`,
`return_home`, `max_bank_deg`, `max_climb_rate`, `max_sink_rate`.

Dostępne kryteria: `no_crash`, `mission_complete`, `max_track_error`, `p95_track_error`,
`mean_track_error`, `max_tilt_deg`, `min_altitude`, `min_battery_soc`, `max_energy_wh`, `min_voltage`,
`max_battery_temp`, `max_landing_error`, `max_nav_error`, `max_saturation_time`, `min_airspeed`,
`min_stall_margin`, `max_airspeed`, `max_stall_time`, `max_load_factor`, `max_altitude_error`,
`min_flight_time`, `max_ice`.

## Własny pojazd

Skopiuj `vehicles/quad_x500.yaml` lub `vehicles/fixed_wing_3kg.yaml` i zmień parametry
(masa, bezwładności, geometria wirników/skrzydła, śmigła, KV, akumulator, pochodne aerodynamiczne).
Najpierw sprawdź go w tunelu: `tunel-aero tunnel vehicles/moj_pojazd.yaml` — raport pokaże
stosunek ciągu do ciężaru, moc zawisu, czas lotu vs temperatura i wysokość (multirotor)
lub CL/CD/Cm, biegunową, trym, prędkość przeciągnięcia i długotrwałość (samolot).

Gotowe pojazdy: `quad_x500` (2 kg, 4S), `hexa_x` (4,5 kg, 6S, redundancja), `quad_mini_250g`
(249 g, kategoria C0), `fixed_wing_3kg` (rozpiętość 2 m).

## Monte Carlo

Odpowiada na pytania typu *„przy jakim wietrze dron przestaje spełniać wymagania?”* —
uruchamia wiele lotów z wylosowanymi warunkami i awariami:

```yaml
base_scenario: ../multirotor/02_wiatr_porywisty_miasto.yaml
runs: 50
randomize:
  environment.wind.speed: {uniform: [2, 14]}
  environment.temperature_c: {normal: [10, 10], min: -20, max: 35}
  environment.wind.terrain: {choice: [suburbs, city]}
  vehicle_overrides.mass: {uniform: [1.9, 2.5]}
random_failures:
  - probability: 0.15
    failure: {type: gps_degraded, t: {uniform: [20, 60]}, duration: {uniform: [10, 30]}, sigma_factor: 4}
```

Raport `montecarlo.html`: odsetek zaliczonych z przedziałem ufności 95%, nałożone trajektorie
(PASS/FAIL), **wykresy wrażliwości** (parametr losowany vs metryka) i rozkłady metryk; dane w CSV.

## Wirtualny tunel aerodynamiczny

```bash
tunel-aero tunnel vehicles/fixed_wing_3kg.yaml --speed 17
tunel-aero tunnel vehicles/quad_x500.yaml --set mass=2.6
```

Z Pythona: `tunel_aero.tunnel.aero_sweep`, `trim_fixed_wing`, `stall_speed`, `fixed_wing_performance`,
`multirotor_hover`, `multirotor_endurance`, `multirotor_forward_power`.

## API Pythona i własne regulatory

```python
from tunel_aero import run_scenario

r = run_scenario("scenarios/multirotor/06_utrata_gps.yaml", seed=3,
                 overrides={"environment.wind.speed": 10})
print(r.passed, r.metrics["max_nav_error"])
r.log["pn"], r.log["alt"], r.log["wind_speed"]   # tablice numpy
```

Własny regulator / estymator / agent RL — interfejs `reset()/step()` jak w Gymnasium:

```python
from tunel_aero.gym_env import FlightEnv

env = FlightEnv("scenarios/multirotor/02_wiatr_porywisty_miasto.yaml",
                action_mode="direct", control_rate=100)
obs, info = env.reset(seed=1)
while True:
    throttles = my_controller(obs["nav"], obs["sensors"])   # 4 wartości 0..1
    obs, reward, terminated, truncated, info = env.step(throttles)
    if terminated or truncated:
        break
metrics, criteria = env.evaluate()
```

`obs["sensors"]` to surowe czujniki (IMU, baro, Pitot, magnetometr, GPS), `obs["nav"]` —
estymata nawigacyjna, `obs["truth"]` (opcjonalnie) — prawdziwy stan do oceny/uczenia.
Działający przykład: `python examples/wlasny_regulator.py`.

## Architektura

```text
tunel_aero/
  cad/               import z CAD: siatki STL/OBJ, przekroje, masa, VLM, raport projektu
  web/ webapi.py webapp.py   aplikacja w przeglądarce (Pyodide) i jej budowa
  atmosphere.py      ISA + temperatura, ciśnienie, wilgotność
  wind.py            profil wiatru (Eurokod), Dryden, podmuchy, mikroburst, termika
  weather.py         deszcz, oblodzenie
  environment.py     złożenie środowiska w każdej chwili
  vehicles/          bryła sztywna 6-DOF, kontakt z ziemią, napęd, akumulator, multirotor, samolot
  sensors.py         czujniki + model błędów nawigacji
  control/           autopiloty multirotora (PX4-like) i samolotu (TECS)
  failures.py        wstrzykiwanie awarii
  sim.py             pętla: środowisko -> awarie -> nawigacja -> autopilot -> aktuatory -> RK4 -> log
  scenario.py        wczytywanie YAML, budowa symulacji
  metrics.py         metryki i kryteria PASS/FAIL
  report.py viewer.py  raport HTML, odtwarzacz 3D
  montecarlo.py tunnel.py gym_env.py cli.py
scenarios/  vehicles/  tests/  examples/
```

Układy współrzędnych: świat NED (północ-wschód-dół, początek w punkcie startu), ciało FRD
(przód-prawo-dół). Kierunek wiatru w konwencji meteorologicznej (skąd wieje).

## Ograniczenia i wiarygodność

To narzędzie do **inżynierskich testów porównawczych i odporności** („co się stanie, jeśli…”),
a nie do certyfikacji. Warto wiedzieć:

- Modele fizyczne opierają się na uznanych źródłach (ISA, Eurokod, MIL-F-8785C, Oseguera-Bowles),
  ale współczynniki deszczu i oblodzenia są półempiryczne — dają poprawne kierunki i rzędy wielkości.
- Parametry gotowych pojazdów są typowe dla swojej klasy, ale nie są pomiarami konkretnego modelu.
  Do decyzji dotyczących Twojej maszyny wprowadź jej dane (najlepiej z pomiarów ciągu i lotów).
- Estymator nawigacji jest modelem błędów („prawda + realistyczny błąd”), a nie pełnym EKF.
  Własny filtr możesz testować na surowych czujnikach przez `FlightEnv`.
- Teren jest płaski (bez przeszkód i budynków — ich wpływ ujmuje szorstkość terenu i turbulencja).
- Model samolotu z CAD to VLM (cienkie powierzchnie, opływ potencjalny) z poprawkami półempirycznymi:
  dobrze oddaje stateczność, sterowność i trym, gorzej — opór i przeciągnięcie (zależą od profilu
  i liczby Reynoldsa, które podajesz w `airfoil`). Pierwsze loty prawdziwego samolotu i tak zaczynaj
  od sprawdzenia wyważenia w zalecanym zakresie.
- Brak podwozia/lądowania samolotu oraz aerodynamiki niestacjonarnej i aeroelastyczności.

Kolejny krok dla pełnej weryfikacji oprogramowania pokładowego: uruchomienie tych samych warunków
z prawdziwym autopilotem w pętli (PX4/ArduPilot SITL), np. przez zastąpienie `control/`
mostem MAVLink.

## Testy

```bash
pytest -q                     # fizyka (ISA, wariancja Drydena, ciągłość mikroburstu, bateria...),
                              # VLM vs teoria, siatki CAD, loty kontrolne, scenariusze, CLI, API, strona
tunel-aero suite --jobs 4     # pełne scenariusze z kryteriami (~75 s)
```
