# planet-hunt: searching NASA TESS data for transits and eclipsing binaries

**Question:** can a first-year student find something new in public NASA TESS light curves, using search
algorithms written out in plain NumPy (no black-box pipelines) so every step can be read and checked?

Faaz Abdulla, BSc (Research) student, Indian Institute of Science, Bengaluru. Semester 1 project (2026).

## Results so far

- **New eclipsing binary: TIC 41602470** (Gaia DR3 2045635134300357504, 19:36:13.52 +33:11:51.6).
  Detached (EA type), period **5.459621 ± 0.000040 d**, eclipses ~7 h long, secondary at phase 0.50.
  First noticed as a diluted signal in the TESS sector 40 light curve of a neighbouring star (TIC 41602482),
  then confirmed in public ZTF g and r data (2018–2025), including a position-shift test showing which of two
  blended stars is the one eclipsing. **Submitted to the AAVSO Variable Star Index (VSX) on 14 Nov 2026.**
  Evidence: [`followup/`](followup/) (folds, periodogram, half-data check, submission draft).
- **Watch list: TIC 138131958**, a single 20-hour, ~770 ppm dip in sector 40. Not a spacecraft artefact; the
  pixel test is unclear. If a dip in sector 79 is the same object, the period would be ~1077 d and the next
  transit ~29 May 2027. Details: [`reports/review_check.md`](reports/review_check.md).
- **Method checks:** recovered WASP-18 b (0.9413 d vs published 0.9415 d) and all three planets of TOI-270
  blind. The single-transit search was tested by planting 300 fake transits in real light curves
  ([`reports/injection_recovery.md`](reports/injection_recovery.md)): it flags about half of them, with far
  fewer false alarms than the first version.

## What each script does

| File | What it does |
|---|---|
| `step1_known_planet.py` | Recovers a known planet (WASP-18 b) to check the basics work |
| `step2_hunt.py` | Box Least Squares period search written from scratch, plus vetting (odd/even depths, secondary eclipse, size) |
| `step3_survey.py` | Periodic search of random sector-40 stars not already listed as planet candidates |
| `step4_single_transits.py` | Single-transit search, 200 stars per run; sorts dips into CANDIDATE / REVIEW / REJECT with every failed check logged; writes reports to `reports/` |
| `step4_followup.py` | Automatic follow-up of interesting dips: ExoFOP, eclipsing-binary catalogues, other TESS sectors, physics limits on the period, ZTF |
| `step4_review_check.py` | Closer look at REVIEW stars: do other stars dip at the same moment, which pixel the dip comes from |
| `step5_followup.py`, `step6_vsx_check.py` | One-star follow-up tests, and independent checks before reporting a new eclipsing binary to VSX |
| `test_step4.py` | Regression tests, known long-period planets, and the injection-recovery test |
| `old/` | The first version of step 4, kept for comparison |

## Running it

Python 3.12+ with `numpy`, `scipy`, `pandas`, `matplotlib`, `requests` and `lightkurve`
(`pip install -r requirements.txt`). Light curves are downloaded from NASA MAST on the fly and cached in
`cache/`, which is not part of this repository. Example: `python step4_single_transits.py`, then
`python test_step4.py all` to rerun the tests (~6 minutes).

## Data

All public: TESS 2-minute light curves (SPOC, sector 40 and others) via MAST, ZTF via IRSA, Gaia DR3,
VSX, ExoFOP and published eclipsing-binary catalogues.

## How this was made

Built with Claude (Anthropic), which wrote the code, designed the analyses and drafted the write-ups.
