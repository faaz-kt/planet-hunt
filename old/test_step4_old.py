"""
Tests for step 4's search, using light curves already saved on this computer
(lightkurve's download cache), so nothing needs downloading.

    1. Wiggliness of every star searched so far: what's normal, what's a pulsator?
    2. The false "PASSES" found on 14 Nov 2026 must NOT pass any more.
    3. Fake transits added to quiet real stars MUST still pass
       (otherwise the new checks are throwing away real planets too).

Run:   python test_step4.py
"""
import csv
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

import step4_single_transits as s4

warnings.filterwarnings("ignore")

CACHE = Path.home() / ".cache" / "lightkurve" / "mastDownload" / "TESS"
FALSE_PASSES = {229742371: "dip at the very start of the sector",
                259271084: "spotted star (VSX: ROT), lopsided dip",
                356602976: "dip only appears with one cleaning setting",
                279916080: "10-hour pulsator",
                399912515: "ragged, triangle-shaped dip"}
DURATIONS = (0.15, 0.3, 0.5)      # fake transit lengths (days)
N_FAKE_STARS = 15
FAKE_SNR = 10                     # how strong the fake transits are


def load(tic):
    """Sector-40 light curve from the cache: time, flux, star radius (same cleaning as step 3)."""
    files = list(CACHE.glob(f"*s{s4.SECTOR:04d}-{tic:016d}*/*lc.fits"))
    if not files:
        return None
    with fits.open(files[0], memmap=False) as h:
        d, hdr = h[1].data, h[0].header
        t, f, q = np.array(d["TIME"], float), np.array(d["PDCSAP_FLUX"], float), np.array(d["QUALITY"])
        radius = hdr.get("RADIUS")
    good = np.isfinite(t) & np.isfinite(f) & (q == 0)
    return t[good], f[good] / np.median(f[good]), radius


def searched_stars():
    with s4.LOG.open(newline="") as fh:
        return [row for row in csv.DictReader(fh) if row.get("n_points")]


def test_wiggliness(rows):
    print("1. Wiggliness of the stars searched so far (1 = pure random noise)")
    values = {}
    for row in rows:
        data = load(int(row["tic"]))
        if data is None or data[0].size < 1000:
            continue
        t, f, _ = data
        flat = s4.hunter.detrend(t, f, np.zeros(t.size, bool), window=s4.WINDOW)
        values[int(row["tic"])] = s4.wiggliness(t, flat)
    w = np.array(list(values.values()))
    print(f"   {w.size} stars: median {np.median(w):.2f}, 90% below {np.percentile(w, 90):.2f}, "
          f"max {w.max():.2f}")
    print(f"   above MAX_WIGGLE = {s4.MAX_WIGGLE}: {np.sum(w > s4.MAX_WIGGLE)} stars "
          f"({np.mean(w > s4.MAX_WIGGLE) * 100:.0f}% would be skipped)")
    for tic in sorted(values, key=values.get, reverse=True)[:8]:
        print(f"      TIC {tic}: {values[tic]:.2f}" + ("  <- the pulsator" if tic == 279916080 else ""))
    return values


def test_false_passes():
    print("\n2. Today's false passes must not pass any more")
    ok = 0
    for tic, why in FALSE_PASSES.items():
        t, f, radius = load(tic)
        res, _ = s4.search(t, f, radius)
        verdict = res["verdict"] if res else "nothing found"
        good = not verdict.startswith("PASSES")
        ok += good
        print(f"   {'OK  ' if good else 'FAIL'} TIC {tic} ({why}): {verdict}")
    print(f"   {ok} of {len(FALSE_PASSES)} correctly rejected")


def test_fake_transits(rows, wiggle):
    print(f"\n3. Fake transits (SNR ~{FAKE_SNR}) in quiet real stars must still pass")
    rng = np.random.default_rng(42)
    # quiet stars that aren't giants (on a giant, an SNR-10 dip means a "planet" bigger than
    # Jupiter, which step 4 rightly rejects - that would test the size check, not the search)
    quiet = [int(r["tic"]) for r in rows if r["verdict"] == "nothing found"
             and wiggle.get(int(r["tic"]), 99) <= s4.MAX_WIGGLE
             and r.get("star_radius") and float(r["star_radius"]) < 2.5]
    picks = rng.choice(quiet, size=min(N_FAKE_STARS, len(quiet)), replace=False)
    results = {D: [] for D in DURATIONS}
    reasons = {}
    for tic in picks:
        t, f, radius = load(int(tic))
        flat = s4.hunter.detrend(t, f, np.zeros(t.size, bool), window=s4.WINDOW)
        tb, fb = s4.hunter.bin_data(t, flat, size=s4.BIN)
        for D in DURATIONS:
            snr, depth = s4.box_snr(tb, fb, D)
            usable = np.where((snr != 0) & (tb > t.min() + 2) & (tb < t.max() - 2))[0]
            noise = s4.robust_noise(depth[snr != 0])
            tc = tb[rng.choice(usable)]
            fake_depth = FAKE_SNR * noise
            # planet-like shape: flat bottom with short slopes in and out
            shape = np.clip((D / 2 - np.abs(t - tc)) / (0.1 * D), 0, 1)
            res, _ = s4.search(t, f * (1 - fake_depth * shape), radius)
            found = res and abs(res.get("time", -99) - tc) < D
            passed = bool(found and res["verdict"].startswith("PASSES"))
            results[D].append(passed)
            if not passed:
                why = res["verdict"] if found else "found a different dip / nothing"
                reasons[why] = reasons.get(why, 0) + 1
    for D, r in results.items():
        print(f"   {D * 24:4.1f} h transits: {sum(r)} of {len(r)} pass")
    total = sum(sum(r) for r in results.values())
    n = sum(len(r) for r in results.values())
    print(f"   overall: {total} of {n} ({total / n * 100:.0f}%)")
    if reasons:
        print("   reasons for misses:")
        for why, k in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"      {k:2d} x {why}")


def without_new_checks():
    """Switch off the checks added on 14 Nov 2026, to see what they cost."""
    s4.dip_survives = lambda *args: (True, [0.0, 0.0, 0.0])
    s4.lopsidedness = lambda *args: 0.0


if __name__ == "__main__":
    rows = searched_stars()
    wiggle = test_wiggliness(rows)
    test_false_passes()
    test_fake_transits(rows, wiggle)
    print("\n   ...and the same fake transits WITHOUT the robustness and symmetry checks:")
    without_new_checks()
    test_fake_transits(rows, wiggle)
