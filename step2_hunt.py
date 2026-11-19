"""
Planet hunt, step 2: a real search, with the algorithm written out in this file.

Improvements over step 1
    - Downloads several TESS sectors and joins them (more dips = fainter planets).
    - Our own BLS search (no black box): see bls_search() below.
    - Protects transits during cleaning: once a dip is found, it is hidden from the
      detrending step and the data is cleaned again, so the dips aren't flattened away.
    - Finds MULTIPLE planets: after finding one, it masks its dips and searches again.
    - Vetting tests on each signal, so you can tell planets from impostors:
        odd/even  : alternating dips with different depths -> two stars eclipsing
        secondary : a dip halfway between transits -> the "planet" glows: likely a star
        size      : bigger than ~2 Jupiters -> probably a star, not a planet

Run:   python step2_hunt.py      (it asks which star; press Enter for the default)
Needs: lightkurve
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_TARGET = "TOI-270"     # try it blind first, then look the star up afterwards!
MAX_SECTORS = 3                # more sectors = fainter planets, but slower downloads
MAX_PLANETS = 3                # how many signals to look for
MIN_SNR = 7.0                  # signals weaker than this are treated as noise
MIN_SDE = 9.0                  # how far the best peak must stand out from all the others
MIN_PERIOD, MAX_PERIOD = 0.5, 20.0       # days
DURATIONS = np.array([0.04, 0.06, 0.08, 0.11, 0.15, 0.2])  # transit lengths to try (days)
BIN = 10 / 60 / 24             # bin the data to 10-minute points for speed

SUN_IN_EARTHS = 109.1
EARTHS_IN_JUPITER = 11.2


# ======================================================================
# Cleaning
# ======================================================================
def detrend(t, f, mask, window=0.5):
    """
    Divide out the star's slow wobbles (spots, pulsations).
    Every `window`/6 days, fit a small parabola to the nearby data and use its value
    as the "trend". A parabola can follow the curve of a wobble far better than a
    plain average. Points where mask is True (known transits) are IGNORED, and points
    far below the fit are thrown out once, so a transit can't drag the trend down and
    erase itself.
    """
    good = ~mask
    grid = np.arange(t.min(), t.max() + window / 6, window / 6)
    trend_at_grid = np.full(grid.size, np.nan)
    for i, g in enumerate(grid):
        near = good & (np.abs(t - g) < window / 2)
        if near.sum() < 20:
            continue
        x, y = t[near] - g, f[near]
        coef = np.polyfit(x, y, 2)
        resid = y - np.polyval(coef, x)
        keep = resid > -3 * np.std(resid)          # drop dips (and other low outliers)
        if keep.sum() >= 20:
            coef = np.polyfit(x[keep], y[keep], 2)
        trend_at_grid[i] = coef[-1]                # the parabola's value at g
    ok = np.isfinite(trend_at_grid)
    trend = np.interp(t, grid[ok], trend_at_grid[ok])
    return f / trend


def bin_data(t, f, size=BIN):
    """Average the data into fixed-width time bins (fewer points = faster search)."""
    idx = np.floor((t - t.min()) / size).astype(int)
    counts = np.bincount(idx)
    keep = counts > 0
    tb = (np.bincount(idx, t) / np.maximum(counts, 1))[keep]
    fb = (np.bincount(idx, f) / np.maximum(counts, 1))[keep]
    return tb, fb


# ======================================================================
# Box Least Squares (BLS): the search itself
# ======================================================================
def bls_search(t, f, periods):
    """
    For every trial period:
      1. fold the data (phase = where in the orbit each point is),
      2. sort it into phase bins,
      3. slide boxes of different widths along the phase,
      4. score each box: depth / (noise / sqrt(points in the box)) = signal-to-noise.
    The best box for the best period is our candidate.
    Returns the SNR for every period, plus the details of the best one.
    """
    dip = 1.0 - f                              # positive where the star is dimmer
    noise = np.std(dip - np.median(dip))
    t_ref = t.min()
    bin_w = DURATIONS.min() / 3                # phase bin width in days
    snr = np.zeros(periods.size)
    best = None

    for i, P in enumerate(periods):
        nb = max(int(P / bin_w), 10)
        b = (((t - t_ref) / P) % 1.0 * nb).astype(int)
        n = np.bincount(b, minlength=nb).astype(float)
        s = np.bincount(b, dip, minlength=nb)
        n2, s2 = np.concatenate([n, n]), np.concatenate([s, s])   # wrap around
        cn, cs = np.concatenate([[0], np.cumsum(n2)]), np.concatenate([[0], np.cumsum(s2)])

        top = 0.0
        for D in DURATIONS:
            k = max(int(round(D / P * nb)), 1)
            n_in = cn[k:k + nb] - cn[:nb]
            s_in = cs[k:k + nb] - cs[:nb]
            with np.errstate(divide="ignore", invalid="ignore"):
                score = np.where(n_in >= 3, s_in / n_in / (noise / np.sqrt(n_in)), 0)
            j = int(np.argmax(score))
            if score[j] > top:
                top = score[j]
                if best is None or top > best["snr"]:
                    centre = (j + k / 2) / nb
                    best = {"snr": top, "period": P, "duration": D,
                            "t0": t_ref + centre * P, "depth": s_in[j] / n_in[j]}
        snr[i] = top
    return snr, best


def period_grid(baseline):
    """Trial periods, evenly spaced in log(period), fine enough not to miss a dip."""
    step = DURATIONS.min() / (3 * min(baseline, 120))
    n = int(np.log(MAX_PERIOD / MIN_PERIOD) / step)
    return np.exp(np.linspace(np.log(MIN_PERIOD), np.log(min(MAX_PERIOD, baseline / 2)), n))


def refine(t, f, cand):
    """Zoom in around the best period for a more precise answer."""
    P = cand["period"]
    fine = np.linspace(P * 0.998, P * 1.002, 400)
    _, best = bls_search(t, f, fine)
    return best if best and best["snr"] >= cand["snr"] else cand


# ======================================================================
# Vetting
# ======================================================================
def in_transit(t, P, t0, D):
    return np.abs(((t - t0 + 0.5 * P) % P) - 0.5 * P) < D / 2


def vet(t, f, c):
    """Simple tests that catch the most common impostors."""
    P, t0, D = c["period"], c["t0"], c["duration"]
    noise = np.std(f)
    epoch = np.round((t - t0) / P).astype(int)
    inside = in_transit(t, P, t0, D * 0.6)        # central part of the dip only

    def depth(sel):
        if sel.sum() < 3:
            return np.nan, np.nan
        return 1 - np.mean(f[sel]), noise / np.sqrt(sel.sum())

    d_odd, e_odd = depth(inside & (epoch % 2 == 1))
    d_even, e_even = depth(inside & (epoch % 2 == 0))
    d_sec, e_sec = depth(in_transit(t, P, t0 + P / 2, D))
    n_transits = len(np.unique(epoch[in_transit(t, P, t0, D)]))

    odd_even_sigma = abs(d_odd - d_even) / np.hypot(e_odd, e_even) if np.isfinite(d_odd + d_even) else 0
    secondary_sigma = d_sec / e_sec if np.isfinite(d_sec) else 0
    return {"n_transits": n_transits, "odd": d_odd, "even": d_even,
            "odd_even_sigma": odd_even_sigma, "secondary_sigma": secondary_sigma}


# ======================================================================
# The whole search
# ======================================================================
def hunt(t, f, max_planets=MAX_PLANETS, verbose=True):
    """Find up to max_planets signals, masking each one before looking for the next."""
    found = []
    mask = np.zeros(t.size, bool)
    baseline = t.max() - t.min()
    periods = period_grid(baseline)

    for n in range(max_planets):
        if verbose:
            print(f"\nSearch {n + 1}: trying {periods.size:,} periods...")
        # 1st pass: clean (protecting dips already found), search
        flat = detrend(t, f, mask)
        tb, fb = bin_data(t[~mask], flat[~mask])
        power, cand = bls_search(tb, fb, periods)
        # SDE: how many "standard deviations" the best peak stands above the rest.
        # Noise always produces SOME peak; a real planet's peak towers over the others.
        sde = (power.max() - power.mean()) / power.std() if power.std() > 0 else 0
        if cand is None or cand["snr"] < MIN_SNR or sde < MIN_SDE:
            if verbose:
                print("  Nothing above the noise. Stopping.")
            break

        # 2nd pass: protect THIS dip too while cleaning, then measure it again
        this = in_transit(t, cand["period"], cand["t0"], cand["duration"] * 1.5)
        flat = detrend(t, f, mask | this)
        tb, fb = bin_data(t[~mask], flat[~mask])
        cand = refine(tb, fb, cand)
        cand["flat"] = flat
        cand["sde"] = sde
        cand.update(vet(t[~mask], flat[~mask], cand))
        found.append(cand)
        if verbose:
            print(f"  Found: period {cand['period']:.4f} d, depth {cand['depth'] * 1e6:.0f} ppm, "
                  f"SNR {cand['snr']:.1f}, SDE {sde:.1f}")

        mask |= in_transit(t, cand["period"], cand["t0"], cand["duration"] * 1.5)
    return found


def verdict(c, star_radius):
    notes = []
    if c["n_transits"] < 2:
        notes.append("only one transit seen")
    if c["odd_even_sigma"] > 3:
        notes.append("odd/even dips differ -> eclipsing stars?")
    if c["secondary_sigma"] > 3:
        notes.append("dip at half-phase -> eclipsing stars?")
    if star_radius:
        r_earth = np.sqrt(max(c["depth"], 0)) * star_radius * SUN_IN_EARTHS
        if r_earth > 2 * EARTHS_IN_JUPITER:
            notes.append("too big for a planet")
    return ("PASSES the basic tests" if not notes else "SUSPICIOUS: " + "; ".join(notes))


# ======================================================================
# Download, run, show
# ======================================================================
def load(target):
    import lightkurve as lk
    results = lk.search_lightcurve(target, mission="TESS", author="SPOC", exptime=120)
    if len(results) == 0:
        raise SystemExit(f"No 2-minute TESS data for '{target}'. Try another star.")
    results = results[:MAX_SECTORS]
    print(f"Downloading {len(results)} sector(s): " + ", ".join(str(s) for s in results.mission))
    lc = results.download_all().stitch().remove_nans()
    lc = lc[lc.quality == 0] if "quality" in lc.colnames else lc
    star_radius = lc.meta.get("RADIUS")
    return lc.time.value, np.asarray(lc.flux.value, float), star_radius


def show(target, t, f, found, star_radius):
    rows = 1 + max(len(found), 1)
    fig, axes = plt.subplots(rows, 1, figsize=(11, 3.2 * rows))
    axes = np.atleast_1d(axes)
    colors = ["red", "dodgerblue", "orange"]

    base = found[0]["flat"] if found else detrend(t, f, np.zeros(t.size, bool))
    axes[0].plot(t, base, ",", color="black", alpha=0.5)
    for c, col in zip(found, colors):
        axes[0].plot(t[in_transit(t, c["period"], c["t0"], c["duration"])],
                     base[in_transit(t, c["period"], c["t0"], c["duration"])], ".", ms=2, color=col)
    axes[0].set_title(f"{target}: cleaned light curve (coloured = transits found)")
    axes[0].set_xlabel("Time (days)")

    for ax, c, col in zip(axes[1:], found, colors):
        phase = ((t - c["t0"] + 0.5 * c["period"]) % c["period"]) - 0.5 * c["period"]
        win = np.abs(phase) < 3 * c["duration"]
        ax.plot(phase[win] * 24, c["flat"][win], ".", ms=1.5, color="lightgray")
        order = np.argsort(phase[win])
        pb, fb = bin_data(phase[win][order], c["flat"][win][order], size=c["duration"] / 8)
        ax.plot(pb * 24, fb, "o", ms=3, color=col)
        ax.set_title(f"Signal: period {c['period']:.4f} d, depth {c['depth'] * 1e6:.0f} ppm, "
                     f"SNR {c['snr']:.1f}  |  {verdict(c, star_radius)}", fontsize=9)
        ax.set_xlabel("Hours from mid-transit")
    fig.tight_layout()
    out = Path(__file__).with_name(f"{target.replace(' ', '_')}_step2.png")
    fig.savefig(out, dpi=140)
    print(f"\nGraph saved to {out}")
    plt.show()


def main():
    target = input(f"Star to search (Enter for {DEFAULT_TARGET}): ").strip() or DEFAULT_TARGET
    t, f, star_radius = load(target)
    f = f / np.nanmedian(f)
    print(f"{t.size:,} measurements over {t.max() - t.min():.1f} days")

    found = hunt(t, f)

    print("\n" + "=" * 70)
    print(f"RESULTS for {target}")
    for i, c in enumerate(found, 1):
        size = ""
        if star_radius:
            r = np.sqrt(max(c["depth"], 0)) * star_radius * SUN_IN_EARTHS
            size = f", about {r:.1f} Earth radii"
        print(f"  Signal {i}: every {c['period']:.4f} days, {c['n_transits']} transits{size}")
        print(f"            {verdict(c, star_radius)}")
    if not found:
        print("  No signals found.")
    print("=" * 70)
    print("Now look the star up on https://exoplanetarchive.ipac.caltech.edu and compare!")
    show(target, t, f, found, star_radius)


if __name__ == "__main__":
    main()
