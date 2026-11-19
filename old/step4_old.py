"""
Planet hunt, step 4: look for SINGLE transits, the ones big pipelines tend to miss.

Why this is where amateurs can win
    A planet with a long year (say 50+ days) may cross its star only ONCE during a
    27-day TESS sector. Period-searching pipelines (and our step 2) need at least two
    dips, so a lone dip often gets ignored. Long-period planets are also the most
    interesting ones: cooler, further out, more like our own solar system.

How the search works
    1. Clean the light curve with a LONGER window than step 2 (lone transits of
       long-period planets last 3-12 hours, and we mustn't erase them).
    2. Slide boxes of different lengths along the light curve (no folding: there is
       nothing to fold) and score each position: depth / (noise / sqrt(points)).
    3. The "upside-down test": run the exact same search for BUMPS instead of dips.
       Bumps can't be planets, so the biggest bump shows how big pure noise and
       glitches get in THIS star. A real dip must clearly beat it.
    4. Vet the best dip: next to a data gap? V-shaped (eclipsing star)? Too deep?
       Does it repeat (then it isn't single, use step 2)?

Run:   python step4_single_transits.py
Needs: lightkurve
"""
import csv
import json
import random
import time
import warnings
import contextlib
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np

import step2_hunt as hunter
import step3_survey as survey

warnings.filterwarnings("ignore", message=".*tpfmodel submodule.*")

SECTOR = 40
N_STARS = 60               # stars per run
PARALLEL = 6               # downloads at the same time (set to 1 if you get errors)
MAX_TMAG = 12.0

WINDOW = 1.5               # cleaning window in days (3x the longest transit we look for)
BIN = 30 / 60 / 24         # 30-minute bins
DURATIONS = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5])   # transit lengths to try (days)
MIN_SNR = 7.5              # noise in tests never got above ~6; real single dips need a clear margin
BUMP_MARGIN = 1.5          # the dip must be this many times stronger than the biggest bump
ROBUST_WINDOWS = (1.5, 2.25, 3.0)  # the dip is re-measured with each of these cleaning windows...
ROBUST_MIN_SNR = 6.0               # ...and must keep this SNR (and half its depth) every time
MAX_WIGGLE = 8.0           # skip stars whose 30-min wiggles are > this many times pure random noise
                           # (typical star ~1.1, 90% below ~5, the 10-hour pulsator 17)
MAX_LOPSIDED = 6.0         # a transit looks the same mirrored; more mismatch than this = not a planet

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "single_transits"
LOG = HERE / "single_log.csv"
CACHE = HERE / "cache"     # saved star lists, so we don't re-download them every run
FIELDS = ["tic", "sector", "tmag", "star_radius", "n_points", "time", "duration_hours",
          "depth_ppm", "snr", "bump_snr", "radius_earths", "period_guess_days", "verdict", "graph",
          "known_as", "wiggle", "robust_snrs"]

# Catalogues of variable stars / eclipsing binaries, checked for every flagged dip.
# VizieR's Harvard mirror (the main Strasbourg server gave errors).
VIZIER = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"
VARIABLE_CATALOGS = {
    "B/vsx/vsx": "VSX",                                   # AAVSO's list of all known variable stars
    "J/ApJS/258/16/tess-ebs": "TESS EB catalogue",        # Prsa et al. 2022, sectors 1-26
    "J/AJ/151/68/catalog": "Kepler EB catalogue",         # Kirk et al. 2016, Kepler field only
}


# ======================================================================
# Search
# ======================================================================
def robust_noise(x):
    """Noise level that ignores outliers (1.4826 x median absolute deviation)."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def box_snr(tb, fb, D, sign=1):
    """
    One box length D, centred on every binned point: depth compared with the data just
    beside it, and SNR = depth / (how much such boxes vary in this star).
    Returns (snr, depth) arrays, or (None, None) if there's too little data.
    """
    cs = np.concatenate([[0], np.cumsum(fb)])

    def mean_between(a, b):
        lo, hi = np.searchsorted(tb, a), np.searchsorted(tb, b)
        n = hi - lo
        with np.errstate(divide="ignore", invalid="ignore"):
            return (cs[hi] - cs[lo]) / n, n

    expected = D / BIN
    m_in, n_in = mean_between(tb - D / 2, tb + D / 2)
    m_left, n_left = mean_between(tb - 1.5 * D, tb - D / 2)
    m_right, n_right = mean_between(tb + D / 2, tb + 1.5 * D)
    ok = (n_in >= 0.8 * expected) & (n_left >= 0.5 * expected) & (n_right >= 0.5 * expected)
    if ok.sum() < 20:
        return None, None
    depth = sign * ((m_left + m_right) / 2 - m_in)
    noise = robust_noise(depth[ok])            # typical size of a "dip" of this length here
    return np.where(ok, depth / noise, 0), depth


def box_scan(tb, fb, sign=1):
    """
    Slide boxes along the (binned) light curve.
    sign=+1 looks for dips, sign=-1 looks for bumps (the upside-down test).

    Each box is compared with the data just BESIDE it (one box-length on each side),
    so slow leftover wobbles don't count. Its strength (SNR) is then measured against
    how much ALL the other boxes of that length in this star vary, so wiggly stars
    aren't judged as if their noise were random point to point.
    Returns events, strongest first: (snr, centre time, duration, depth).
    """
    events = []
    for D in DURATIONS:
        snr, depth = box_snr(tb, fb, D, sign)
        if snr is None:
            continue
        for i in np.argsort(snr)[::-1][:20]:
            if snr[i] > 0:
                events.append((float(snr[i]), float(tb[i]), float(D), float(depth[i])))
    # keep the strongest, drop weaker events that overlap it
    events.sort(reverse=True)
    kept = []
    for e in events:
        if all(abs(e[1] - k[1]) > (e[2] + k[2]) / 2 for k in kept):
            kept.append(e)
    return kept


def shape_test(tb, fb, tc, D):
    """
    Fit two shapes to the dip and see which fits better:
      a planet shape (flat bottom, short slopes in and out)  vs  a V (two stars grazing).
    Returns how much better the V fits, in units of 'noise squared' (> 9 = clearly a V).
    """
    near = np.abs(tb - tc) < 2.5 * D
    outside = near & (np.abs(tb - tc) > 0.75 * D)
    if outside.sum() < 5:
        return 0.0
    y = np.median(fb[outside]) - fb[near]
    x = tb[near] - tc
    sigma2 = robust_noise(fb[outside]) ** 2 or 1e-12

    def best_chi2(profile):
        best = np.inf
        for w in D * np.array([0.6, 0.8, 1.0, 1.25, 1.5, 2.0]):
            for shift in D * np.array([-0.1, 0, 0.1]):
                s = profile(np.abs(x - shift) / (w / 2))
                if s @ s > 0:
                    a = (s @ y) / (s @ s)
                    best = min(best, np.sum((y - a * s) ** 2))
        return best

    planet = best_chi2(lambda u: np.clip((1 - u) / 0.15, 0, 1))
    vee = best_chi2(lambda u: np.clip(1 - u, 0, 1))
    return (planet - vee) / sigma2


def lopsidedness(tb, fb, tc, D):
    """
    A planet's transit is symmetric: the way in mirrors the way out. Compare the right
    half of the dip with the mirrored left half (trying small shifts of the centre).
    Returns the mismatch in units of the noise: ~1 for a symmetric dip. A spotted star's
    "dip" with a sharp drop and slow recovery (TIC 259271084) scores far higher.
    A mismatch of 10% of the depth is allowed for free: with 30-minute bins the dip's
    edges never line up perfectly, and for deep dips that alone would look "lopsided".
    """
    depth = np.median(fb[np.abs(tb - tc) > D]) - np.median(fb[np.abs(tb - tc) < 0.3 * D]) \
        if np.any(np.abs(tb - tc) < 0.3 * D) else 0.0
    sigma2 = (robust_noise(np.diff(fb)) / np.sqrt(2)) ** 2 + (0.1 * depth) ** 2 or 1e-12
    best = np.inf
    for shift in D * np.linspace(-0.25, 0.25, 11):
        x = tb - (tc + shift)
        left, right = (np.abs(x) < D) & (x < 0), (np.abs(x) < D) & (x > 0)
        if left.sum() < 3 or right.sum() < 3:
            continue
        order = np.argsort(-x[left])
        mirrored = np.interp(x[right], -x[left][order], fb[left][order])
        best = min(best, np.mean((fb[right] - mirrored) ** 2) / (2 * sigma2))
    return best if np.isfinite(best) else 0.0


def vet_event(tb, fb, ev, others, star_radius):
    """Checks that catch glitches and eclipsing stars. Returns a list of problems."""
    snr, tc, D, depth = ev
    problems = []

    # 1. data on both sides (glitches love the edges of data gaps)
    for a, b in ((tc - 1.5 * D, tc - 0.5 * D), (tc + 0.5 * D, tc + 1.5 * D)):
        if np.sum((tb > a) & (tb < b)) < 0.5 * D / BIN:
            problems.append("next to a data gap")
            break

    # 2. shape: a planet's dip is flat-bottomed (U), two stars give a V
    if shape_test(tb, fb, tc, D) > 9:
        problems.append("V-shaped")
    lop = lopsidedness(tb, fb, tc, D)
    if lop > MAX_LOPSIDED:
        problems.append(f"lopsided ({lop:.1f})")

    # 3. size
    if depth > 0.03:
        problems.append("very deep (star?)")
    if star_radius:
        r_earth = np.sqrt(max(depth, 0)) * star_radius * hunter.SUN_IN_EARTHS
        if r_earth > 2 * hunter.EARTHS_IN_JUPITER:
            problems.append("too big for a planet")

    # 4. does it repeat? (then it isn't a single transit)
    similar = [o for o in others if o[0] >= MIN_SNR * 0.6 and 0.6 < o[3] / depth < 1.6
               and abs(o[2] - D) <= 0.1]
    if similar:
        problems.append(f"repeats ({len(similar) + 1} similar dips): use step 2")
    return problems


def wiggliness(t, flat):
    """
    How much the cleaned light curve wiggles on 30-minute-to-day timescales, compared
    with what pure random noise would give. ~1 for a quiet star; a pulsating star
    (like TIC 279916080, which goes up and down every ~10 h) scores much higher, and
    its wiggles can line up into fake dips.
    """
    tb, fb = hunter.bin_data(t, flat, size=BIN)
    point_noise = robust_noise(np.diff(flat)) / np.sqrt(2)      # random noise of one point
    points_per_bin = BIN / np.median(np.diff(t))
    return robust_noise(fb) / (point_noise / np.sqrt(points_per_bin))


def dip_survives(t, f, tc, D):
    """
    Re-measure the dip after cleaning with several window lengths. A real dip keeps
    its depth and SNR; a dip made by the cleaning, or by the star's own wiggles, changes
    a lot (TIC 356602976: SNR 10.3 / 5.3 / 3.9). Returns (passes?, list of SNRs).
    """
    mask = np.abs(t - tc) < 0.75 * D
    snrs, depths = [], []
    for window in ROBUST_WINDOWS:
        flat = hunter.detrend(t, f, mask, window=window)
        tb, fb = hunter.bin_data(t, flat, size=BIN)
        snr, depth = box_snr(tb, fb, D)
        j = np.argmin(np.abs(tb - tc))
        snrs.append(float(snr[j]) if snr is not None else 0.0)
        depths.append(float(depth[j]) if snr is not None else 0.0)
    passes = min(snrs) >= ROBUST_MIN_SNR and min(depths) >= 0.5 * max(depths)
    return passes, snrs


def period_guess(duration_days, star_radius):
    """
    Very rough orbital period from the transit length, for a central transit:
        duration ~ 13 h x (P / 365 d)^(1/3) x (star density / Sun's)^(-1/3)
    with density ~ 1 / R^2 for normal stars. Off-centre transits are shorter,
    so the real period is probably LONGER than this.
    """
    density = 1 / star_radius ** 2 if star_radius else 1.0
    return 365.25 * (duration_days * 24 / 13) ** 3 * density


def search(t, f, star_radius):
    """Full single-transit search on one light curve. Returns (result dict, flat)."""
    flat = hunter.detrend(t, f, np.zeros(t.size, bool), window=WINDOW)
    wiggle = round(float(wiggliness(t, flat)), 2)
    if wiggle > MAX_WIGGLE:
        return {"verdict": f"skipped: too variable (wiggles {wiggle}x random noise)", "wiggle": wiggle}, flat
    tb, fb = hunter.bin_data(t, flat, size=BIN)
    dips = box_scan(tb, fb, +1)
    if not dips:
        return None, flat

    # Protect a dip and clean again, so it isn't partly erased. The first cleaning can
    # weaken a long dip enough that it isn't the top one yet, so try the top 3 and keep
    # whichever comes out strongest.
    tries = []
    for cand in dips[:3]:
        mask = np.abs(t - cand[1]) < 0.75 * cand[2]
        flat_c = hunter.detrend(t, f, mask, window=WINDOW)
        tb_c, fb_c = hunter.bin_data(t, flat_c, size=BIN)
        dips_c = box_scan(tb_c, fb_c, +1)
        if dips_c:
            tries.append((dips_c[0][0], flat_c, tb_c, fb_c, dips_c))
    if not tries:
        return None, flat
    _, flat, tb, fb, dips = max(tries, key=lambda x: x[0])
    best = dips[0]
    away = np.abs(tb - best[1]) > 2 * best[2]          # leave the dip itself out
    bumps = box_scan(tb[away], fb[away], -1)
    bump_snr = bumps[0][0] if bumps else 0.0

    snr, tc, D, depth = best
    result = {"time": round(tc, 3), "duration_hours": round(D * 24, 1),
              "depth_ppm": round(depth * 1e6), "snr": round(snr, 1), "bump_snr": round(bump_snr, 1),
              "radius_earths": None, "period_guess_days": None, "wiggle": wiggle}
    if star_radius:
        result["radius_earths"] = round(np.sqrt(max(depth, 0)) * star_radius * hunter.SUN_IN_EARTHS, 1)
        result["period_guess_days"] = round(period_guess(D, star_radius))

    if snr < MIN_SNR:
        result["verdict"] = "nothing found"
    elif snr < BUMP_MARGIN * bump_snr:
        result["verdict"] = "nothing found (dip no bigger than the noise bumps)"
    else:
        problems = vet_event(tb, fb, best, dips[1:], star_radius)
        survives, snrs = dip_survives(t, f, tc, D)
        result["robust_snrs"] = "/".join(f"{s:.1f}" for s in snrs)
        if not survives:
            problems.append(f"changes with cleaning (SNR {result['robust_snrs']})")
        result["verdict"] = ("PASSES the basic tests" if not problems
                             else "SUSPICIOUS: " + "; ".join(problems))
    result["_event"] = best
    return result, flat


# ======================================================================
# Plot
# ======================================================================
def save_plot(tic, sector, t, flat, res, out):
    import matplotlib.pyplot as plt
    snr, tc, D, depth = res["_event"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    tb, fb = hunter.bin_data(t, flat, size=BIN)
    axes[0].plot(t, flat, ",", color="black", alpha=0.4)
    axes[0].plot(tb, fb, ".", ms=2, color="gray")
    axes[0].axvspan(tc - D, tc + D, color="red", alpha=0.15)
    axes[0].set_title(f"TIC {tic}, sector {sector}: whole sector (red = the event)")
    axes[0].set_xlabel("Time (days)")

    win = np.abs(t - tc) < 4 * D
    winb = np.abs(tb - tc) < 4 * D
    axes[1].plot((t[win] - tc) * 24, flat[win], ".", ms=2, color="lightgray")
    axes[1].plot((tb[winb] - tc) * 24, fb[winb], "o", ms=4, color="red")
    axes[1].axhline(1 - depth, color="red", lw=0.8, ls="--")
    size = f", ~{res['radius_earths']} Earth radii" if res["radius_earths"] else ""
    guess = f", period maybe {res['period_guess_days']}+ days" if res["period_guess_days"] else ""
    axes[1].set_title(f"Zoom: {res['duration_hours']} h long, {res['depth_ppm']} ppm deep{size}{guess}\n"
                      f"SNR {res['snr']} (biggest bump: {res['bump_snr']})  |  {res['verdict']}",
                      fontsize=9)
    axes[1].set_xlabel("Hours from the middle of the event")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ======================================================================
# Survey
# ======================================================================
def fetch(tic):
    """
    Runs in a separate helper process: download only (the slow part).
    Separate PROCESSES, not threads: the download library swaps the screen output
    around while it works, and with threads that crashed our printing.
    Its "Downloading URL..." messages are hidden to keep the output readable.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return tic, survey.download(tic, SECTOR), None
    except Exception as e:
        return tic, None, f"{e.__class__.__name__}: {e}"[:120]


def cached(name, make, max_age_days):
    """
    Return a saved list of TIC numbers if it's recent enough, otherwise build it with
    make() and save it. Saves ~1.5 minutes per run (those lists are slow to download).
    Delete the cache folder to force fresh lists.
    """
    path = CACHE / f"{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        ids = set(json.loads(path.read_text()))
        print(f"  using saved list {path.name} ({len(ids):,} stars)")
        return ids
    ids = make()
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(sorted(ids)))
    return ids


def pick_bright_stars(fresh, n):
    """
    Choose n random stars that are bright enough, checking brightness BEFORE downloading
    (one quick catalogue query per 150 stars). Last run, 22 of 60 downloads were wasted
    on stars that were then skipped as too faint.
    Returns {tic: (ra, dec)}; the position is used later for the catalogue check.
    """
    from astroquery.mast import Catalogs
    fresh = list(fresh)
    random.shuffle(fresh)
    picks = {}
    for start in range(0, len(fresh), 150):
        if len(picks) >= n:
            break
        table = Catalogs.query_criteria(catalog="TIC", ID=fresh[start:start + 150])
        for row in table:
            if len(picks) < n and not np.ma.is_masked(row["Tmag"]) and row["Tmag"] <= MAX_TMAG:
                picks[int(row["ID"])] = (float(row["ra"]), float(row["dec"]))
    return picks


def known_variable(ra, dec):
    """
    Is there already a known variable star / eclipsing binary at this position?
    Asks VSX, the TESS EB catalogue and the Kepler EB catalogue in one query (5" radius).
    Returns e.g. "VSX type EA, P=3.8994 d" or "not in catalogues".
    """
    import requests
    try:
        text = requests.get(VIZIER, timeout=30, params={
            "-source": " ".join(VARIABLE_CATALOGS), "-c": f"{ra} {dec:+}", "-c.rs": 5,
            "-out.max": 5}).text
    except Exception as e:
        return f"catalogue check failed ({e.__class__.__name__})"
    # The answer is one table per catalogue: "#Name: <catalogue>", a header line,
    # a units line, a line of dashes, then the matching stars.
    found, table, header, in_data = [], None, None, False
    for line in text.splitlines():
        if line.startswith("#Name:"):
            table, header, in_data = line.split(":", 1)[1].strip(), None, False
        elif line.startswith("#") or not line.strip():
            in_data = False
        elif header is None:
            header = [h.strip() for h in line.split("\t")]
        elif line.startswith("-"):
            in_data = True
        elif in_data and table in VARIABLE_CATALOGS:
            row = dict(zip(header, (v.strip() for v in line.split("\t"))))
            label = VARIABLE_CATALOGS[table]
            if row.get("Type"):
                label += f" type {row['Type']}"
            period = row.get("Period") or row.get("Per")
            if period:
                label += f", P={float(period):.4f} d"
            found.append(label)
    return "; ".join(found) if found else "not in catalogues"


def upgrade_log():
    """Older logs miss the newer columns (known_as, wiggle...); rewrite the file with them."""
    if not LOG.exists():
        return
    with LOG.open(newline="") as f:
        reader = csv.DictReader(f)
        if set(FIELDS) <= set(reader.fieldnames or []):
            return
        rows = list(reader)
    with LOG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def analyse(tic, data, position=None):
    base = {"tic": tic, "sector": SECTOR}
    if data is None:
        return dict(base, verdict="no data")
    t, f, tmag, radius = data
    base.update(tmag=tmag, star_radius=radius, n_points=t.size)
    if tmag is not None and tmag > MAX_TMAG:
        return dict(base, verdict=f"skipped: too faint (Tmag {tmag:.1f})")
    if t.size < 1000:
        return dict(base, verdict="skipped: too little data")
    res, flat = search(t, f, radius)
    if res is None:
        return dict(base, verdict="nothing found")
    row = dict(base, **{k: v for k, v in res.items() if not k.startswith("_")})
    if not res["verdict"].startswith(("nothing", "skipped")):
        out = OUT_DIR / f"TIC_{tic}_S{SECTOR}_single.png"
        save_plot(tic, SECTOR, t, flat, res, out)
        row["graph"] = out.name
        if position:
            row["known_as"] = known_variable(*position)
    return row


def main():
    matplotlib.use("Agg")                      # save graphs without opening windows
    OUT_DIR.mkdir(exist_ok=True)
    upgrade_log()
    stars = cached(f"sector{SECTOR}_targets", lambda: survey.sector_targets(SECTOR), 365)
    print("Removing stars that already have planets or candidates:")
    known = cached("known_stars", survey.known_stars, 3)
    fresh = stars - known - survey.already_logged(LOG)
    print(f"{len(fresh):,} stars to choose from.")
    print(f"Choosing {N_STARS} bright enough ones (Tmag <= {MAX_TMAG})...")
    positions = pick_bright_stars(fresh, N_STARS)
    picks = list(positions)
    print()

    start, interesting = time.time(), []
    print(f"Searching {len(picks)} stars, {PARALLEL} downloads at a time...")
    with ProcessPoolExecutor(max_workers=PARALLEL) as pool:
        jobs = [pool.submit(fetch, tic) for tic in picks]
        for n, job in enumerate(as_completed(jobs), 1):
            tic, data, err = job.result()
            try:
                row = analyse(tic, data, positions[tic]) if err is None else {
                    "tic": tic, "sector": SECTOR, "verdict": f"error: {err}"}
            except Exception as e:
                row = {"tic": tic, "sector": SECTOR, "verdict": f"error: {e.__class__.__name__}"}
            survey.write_log([row], LOG, FIELDS)
            detail = f"{row['depth_ppm']} ppm, {row['duration_hours']} h, SNR {row['snr']}: " if row.get("snr") \
                and not row["verdict"].startswith("nothing") else ""
            known_as = f"\n      catalogues: {row['known_as']}" if row.get("known_as") else ""
            print(f"[{n}/{len(picks)}] TIC {tic}: {detail}{row['verdict']}{known_as}", flush=True)
            if row["verdict"].startswith("PASSES"):
                interesting.append(row)

    print("\n" + "=" * 70)
    print(f"{len(picks)} stars in {(time.time() - start) / 60:.1f} minutes.")
    if interesting:
        print("Single dips that PASS the basic tests (graphs in the single_transits folder):")
        for r in interesting:
            print(f"  TIC {r['tic']}: {r['depth_ppm']} ppm, {r['duration_hours']} h, "
                  f"~{r['radius_earths']} Earth radii -> {r['graph']}  [{r.get('known_as', '')}]")
        print("\nOpen each graph. If it looks like a clean, flat-bottomed dip, check the TIC")
        print("on https://exofop.ipac.caltech.edu/tess/ and then show me!")
    else:
        print("No single transits passed this time. Run again for more stars.")
    print(f"Full log: {LOG}")


if __name__ == "__main__":
    main()
