"""
Planet hunt, step 3: a mini survey of stars nobody has flagged yet.

What it does
    1. Gets the list of stars TESS watched with 2-minute data in one sector.
    2. Removes every star that already has a known planet or a planet candidate
       (TOIs and community candidates from ExoFOP, plus NASA's planet list).
    3. Picks some of the rest at random and runs the step 2 hunter on each.
    4. Writes everything to survey_log.csv and saves a graph for every signal,
       in the "survey" folder. Stars already in the log are never searched twice,
       so you can run this again and again to keep going.

Most stars will show nothing. Most signals that DO show up will be impostors
(eclipsing stars, telescope glitches). That's normal: it's what real surveys see.

Run:   python step3_survey.py
Needs: lightkurve (astroquery and pandas come with it)
"""
import csv
import random
import warnings
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import step2_hunt as hunter

# lightkurve warns about an optional extra we don't use; hide it
warnings.filterwarnings("ignore", message=".*tpfmodel submodule.*")

SECTOR = 40                # which TESS sector to search (any from 1 to ~90)
N_STARS = 20               # how many stars per run
MAX_TMAG = 12.0            # skip stars fainter than this (too noisy). Smaller = brighter.

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "survey"
LOG = HERE / "survey_log.csv"
LOG_FIELDS = ["tic", "sector", "tmag", "star_radius", "n_points", "signal",
              "period_days", "depth_ppm", "snr", "sde", "n_transits",
              "radius_earths", "verdict", "graph"]

KNOWN_SOURCES = {
    "TOIs (ExoFOP)": "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv",
    "community candidates (ExoFOP)": "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv",
    "confirmed planets (NASA)": ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
                                 "select+distinct+tic_id+from+pscomppars+where+tic_id+is+not+null&format=csv"),
}


# ----------------------------------------------------------------------
# Lists of stars
# ----------------------------------------------------------------------
def sector_targets(sector):
    """TIC numbers of all stars with 2-minute SPOC light curves in this sector."""
    from astroquery.mast import Observations
    print(f"Asking NASA for the star list of sector {sector} (can take a minute)...")
    obs = Observations.query_criteria(obs_collection="TESS", dataproduct_type="timeseries",
                                      provenance_name="SPOC", sequence_number=sector)
    tics = set()
    for name in obs["target_name"]:
        if str(name).isdigit():
            tics.add(int(name))
    print(f"  {len(tics):,} stars")
    return tics


def known_stars():
    """TIC numbers of stars that already have a planet or a candidate."""
    known = set()
    for label, url in KNOWN_SOURCES.items():
        try:
            table = pd.read_csv(url, comment="#", dtype=str)
            col = next(c for c in table.columns if "tic" in c.lower().replace(" ", "_"))
            ids = {int(m) for v in table[col].dropna() for m in re.findall(r"\d+", v)[:1]}
            known |= ids
            print(f"  {label}: {len(ids):,} stars")
        except Exception as e:
            print(f"  Couldn't load {label} ({e.__class__.__name__}). "
                  f"Check any signal on ExoFOP yourself.")
    return known


def already_logged(log=None):
    """Stars searched in earlier runs (from the log file and any backup logs)."""
    log = log or LOG
    done = set()
    for path in log.parent.glob(f"{log.stem}*.csv"):
        try:
            with path.open(newline="") as f:
                done |= {int(row["tic"]) for row in csv.DictReader(f) if row.get("tic")}
        except OSError:
            pass
    return done


def write_log(rows, log=None, fields=None):
    """
    Add rows to the log (survey_log.csv by default). If the file is locked (open in Excel, or OneDrive is
    syncing it), wait a little and try again; if it stays locked, write a backup log
    instead of crashing.
    """
    log, fields = log or LOG, fields or LOG_FIELDS
    for attempt in range(5):
        try:
            new_file = not log.exists()
            with log.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if new_file:
                    w.writeheader()
                w.writerows(rows)
            return
        except PermissionError:
            time.sleep(2)
    backup = log.with_name(f"{log.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with backup.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  ({log.name} is locked - is it open in Excel? Saved to {backup.name} instead)")


# ----------------------------------------------------------------------
# One star
# ----------------------------------------------------------------------
def download(tic, sector):
    import lightkurve as lk
    res = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", exptime=120, sector=sector)
    if len(res) == 0:
        return None
    lc = res[0].download().remove_nans()
    if "quality" in lc.colnames:
        lc = lc[lc.quality == 0]
    f = np.asarray(lc.flux.value, float)
    return lc.time.value, f / np.nanmedian(f), lc.meta.get("TESSMAG"), lc.meta.get("RADIUS")


def search_star(tic, sector):
    base = {"tic": tic, "sector": sector}
    data = download(tic, sector)
    if data is None:
        return [dict(base, verdict="no data")]
    t, f, tmag, radius = data
    base.update(tmag=tmag, star_radius=radius, n_points=t.size)
    if tmag is not None and tmag > MAX_TMAG:
        return [dict(base, verdict=f"skipped: too faint (Tmag {tmag:.1f})")]
    if t.size < 1000:
        return [dict(base, verdict="skipped: too little data")]

    found = hunter.hunt(t, f, max_planets=2, verbose=False)
    if not found:
        return [dict(base, verdict="nothing found")]

    graph = OUT_DIR / f"TIC_{tic}_S{sector}.png"
    hunter.show(f"TIC {tic}", t, f, found, radius, out=graph, display=False)
    rows = []
    for i, c in enumerate(found, 1):
        r = np.sqrt(max(c["depth"], 0)) * radius * hunter.SUN_IN_EARTHS if radius else None
        rows.append(dict(base, signal=i, period_days=round(c["period"], 5),
                         depth_ppm=round(c["depth"] * 1e6), snr=round(c["snr"], 1),
                         sde=round(c["sde"], 1), n_transits=c["n_transits"],
                         radius_earths=round(r, 2) if r else None,
                         verdict=hunter.verdict(c, radius), graph=graph.name))
    return rows


# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    stars = sector_targets(SECTOR)
    print("Removing stars that already have planets or candidates:")
    fresh = sorted(stars - known_stars() - already_logged())
    print(f"{len(fresh):,} stars nobody has flagged (that we know of).\n")
    if not fresh:
        return

    picks = random.sample(fresh, min(N_STARS, len(fresh)))
    interesting = []
    for n, tic in enumerate(picks, 1):
        start = time.time()
        print(f"[{n}/{len(picks)}] TIC {tic} ... ", end="", flush=True)
        try:
            rows = search_star(tic, SECTOR)
        except Exception as e:
            rows = [{"tic": tic, "sector": SECTOR, "verdict": f"error: {e.__class__.__name__}"}]
        write_log(rows)
        summary = "; ".join(
            f"{r['period_days']} d, {r['depth_ppm']} ppm, {r['verdict']}" if r.get("period_days") else r["verdict"]
            for r in rows)
        print(f"{summary}  ({time.time() - start:.0f}s)")
        interesting += [r for r in rows if str(r.get("verdict", "")).startswith("PASSES")]

    print("\n" + "=" * 70)
    if interesting:
        print("Signals that PASS the basic tests (look at their graphs in the survey folder):")
        for r in interesting:
            print(f"  TIC {r['tic']}: every {r['period_days']} d, {r['depth_ppm']} ppm, "
                  f"SNR {r['snr']}, about {r['radius_earths']} Earth radii -> {r['graph']}")
        print("\nFor each one, search the TIC number on https://exofop.ipac.caltech.edu/tess/")
        print("If nobody has reported it, show me the graph!")
    else:
        print("No signals passed the tests this time. Run it again for 20 more stars.")
    print(f"Full log: {LOG}")


if __name__ == "__main__":
    main()
