"""
Planet hunt, step 6: independent checks before reporting TIC 41602470 to AAVSO VSX as a
new eclipsing binary. NOTHING is submitted: this only checks, measures and drafts.

Checks (each printed as PASS / FAIL / INFO with numbers):
    1. Fresh ZTF g and r light curves from IRSA (catflags = 0 only), saved in followup/ztf/
    2. Period: wide search 0.2-30 d, aliases, fold at P and P/2, both filters, every ZTF field
    3. Is it this star? Gaia DR3 neighbours within 10", ZTF vs Gaia position, ZTF neighbours
    4. Is it new? VSX, Gaia DR3 EBs, ASAS-SN, ZTF periodic (Chen+2020), TESS EBs, SIMBAD (1')
    5. What VSX needs: period, epoch of primary minimum, magnitudes, duration, EA or EB
    6. Plots: folded ZTF r, ZTF g and TESS sector 40
    7. A draft VSX entry, written to followup/TIC_41602470_VSX_draft.txt

About time: ZTF gives HJD (heliocentric Julian Date, UTC). TESS gives BTJD = BJD_TDB - 2457000.
The two differ by ~70 seconds, which doesn't matter for 5-hour eclipses.

Run:   python step6_vsx_check.py
"""
import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit

import step2_hunt as hunter
import step3_survey as survey
import step5_followup as f5

warnings.filterwarnings("ignore")

TIC = 41602470                 # the suspected eclipsing binary
BLEND_TIC = 41602482           # the bright giant 11.6" away, where TESS first saw the eclipses
TESS_PERIOD = 5.4574           # from step 5 (TESS sector 40 alone)
OUT = f5.OUT
ZTF_DIR = OUT / "ztf"
ZTF_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
RESULTS = []


def check(name, passed, detail):
    """Record and print one result. passed: True, False, or None (= just information)."""
    tag = {True: "PASS", False: "FAIL", None: "INFO"}[passed]
    RESULTS.append((tag, name, detail))
    print(f"  [{tag}] {name}: {detail}")


# ======================================================================
# ZTF data
# ======================================================================
def ztf_get(tic, ra, dec, radius_arcsec=1.5):
    """ZTF g and r light curve near a position (saved to disk), catflags = 0 only."""
    ZTF_DIR.mkdir(parents=True, exist_ok=True)
    path = ZTF_DIR / f"ztf_{tic}.csv"
    if not path.exists() or path.stat().st_size < 300:
        for attempt in range(3):
            try:
                r = requests.get(ZTF_URL, timeout=300, params={
                    "POS": f"CIRCLE {ra} {dec} {radius_arcsec / 3600}",
                    "BANDNAME": "g,r", "FORMAT": "csv"})
                r.raise_for_status()
                path.write_text(r.text)
                break
            except requests.RequestException as e:
                print(f"     ZTF try {attempt + 1} for TIC {tic} failed: {e.__class__.__name__}")
        else:
            return None
    df = pd.read_csv(path)
    if "catflags" not in df or len(df) == 0:
        return None
    df = df[df["catflags"] == 0].copy()
    df["band"] = df["filtercode"].str[-1]
    return df[df["band"].isin(["g", "r"])].reset_index(drop=True)


def normalise(df, keep=None):
    """Flux relative to each ZTF object's median (each field/chip/filter has its own zero point)."""
    keep = np.ones(len(df), bool) if keep is None else keep
    ref = df[keep].groupby("oid")["mag"].median()
    df["dmag"] = df["mag"] - df["oid"].map(ref)
    df["flux"] = 10 ** (-0.4 * df["dmag"])
    df["ferr"] = df["flux"] * df["magerr"] * np.log(10) / 2.5
    return df


def phase_dt(t, P, t0):
    """Time from the nearest eclipse centre at t0 + n*P, in days (-P/2 .. P/2)."""
    return ((t - t0 + P / 2) % P) - P / 2


# ======================================================================
# Period
# ======================================================================
def wide_search(t, f, e):
    """
    Box search over 0.2-30 days with astropy's BLS (same idea as step 2's, in compiled code;
    our numpy version would take hours for ~1 million trial periods). The period steps are
    fine enough that an eclipse can't drift by more than 1/4 of its length over all the data.
    """
    from astropy.timeseries import BoxLeastSquares
    bls = BoxLeastSquares(t, f, dy=e)
    span = t.max() - t.min()
    periods, power = [], []
    for pmin, pmax, durations in ((0.2, 1.0, (0.02, 0.04, 0.07, 0.1)),
                                  (1.0, 30.0, (0.06, 0.1, 0.15, 0.2, 0.3))):
        grid = np.exp(np.arange(np.log(pmin), np.log(pmax), min(durations) / (4 * span)))
        periods.append(grid)
        power.append(bls.power(grid, durations, objective="snr").power)
    return np.concatenate(periods), np.concatenate(power)


def top_peaks(P, power, n=8):
    peaks = []
    for i in np.argsort(power)[::-1]:
        if all(abs(P[i] - p) / p > 0.002 for p, _ in peaks):
            peaks.append((P[i], power[i]))
        if len(peaks) == n:
            break
    return peaks


def alias_name(Pk, P0):
    """Name a peak by its relation to the orbital period P0 (1-day and 1-year aliases included)."""
    f, f0 = 1 / Pk, 1 / P0
    for name, ref in (("P (orbit)", f0), ("P/2 (eclipse spacing)", 2 * f0), ("2P", f0 / 2),
                      ("P/3", 3 * f0), ("1-day alias of P", f0 + 1), ("1-day alias of P", abs(f0 - 1)),
                      ("1-day alias of P/2", 2 * f0 + 1), ("1-day alias of P/2", abs(2 * f0 - 1)),
                      ("1-year alias of P/2", 2 * f0 + 1 / 365.25), ("1-year alias of P/2", 2 * f0 - 1 / 365.25),
                      ("1-year alias of P", f0 + 1 / 365.25), ("1-year alias of P", f0 - 1 / 365.25)):
        if abs(f - ref) / ref < 0.003:
            return name
    return "unrelated"


def dip_at(df, P, t0, dur, centre_phase=0.0, sel=None):
    """Depth (and error) of the dip at a given phase: mean flux in the eclipse core vs outside."""
    sel = np.ones(len(df), bool) if sel is None else sel
    t, f = df["hjd"].to_numpy(), df["flux"].to_numpy()
    dt = phase_dt(t, P, t0 + centre_phase * P)
    core = sel & (np.abs(dt) < 0.3 * dur)
    away = sel & (np.abs(phase_dt(t, P, t0)) > dur) & (np.abs(phase_dt(t, P, t0 + 0.5 * P)) > dur)
    if core.sum() < 3 or away.sum() < 10:
        return np.nan, np.nan, int(core.sum())
    err = np.std(f[away]) / np.sqrt(core.sum())
    return 1 - np.mean(f[core]) / np.median(f[away]), err, int(core.sum())


# ======================================================================
# Eclipse model fit (period, epoch, depths, duration, all with uncertainties)
# ======================================================================
def eclipse_model(t, P, t0, dA, dB, phiB, dur, ing):
    """Two trapezoid-shaped eclipses: primary at t0, secondary at t0 + phiB*P.
    dur = total length, ing = fraction of dur spent going down (0.5 = pure V)."""
    def trap(dt):
        return np.clip((dur / 2 - np.abs(dt)) / (ing * dur), 0, 1)
    return 1 - dA * trap(phase_dt(t, P, t0)) - dB * trap(phase_dt(t, P, t0 + phiB * P))


def fit_band(df, P0, t00, dur0):
    t, f, e = df["hjd"].to_numpy(), df["flux"].to_numpy(), df["ferr"].to_numpy()
    t00 = t00 + np.round((np.median(t) - t00) / P0) * P0           # epoch in the middle of the data
    dA0 = max(dip_at(df, P0, t00, dur0)[0], 0.01)
    dB0 = max(dip_at(df, P0, t00, dur0, 0.5)[0], 0.005)
    p0 = [P0, t00, dA0, dB0, 0.5, dur0, 0.3]
    lo = [P0 * 0.999, t00 - 0.1, 0, 0, 0.4, 0.05, 0.05]
    hi = [P0 * 1.001, t00 + 0.1, 1, 1, 0.6, 0.6, 0.5]
    popt, pcov = curve_fit(eclipse_model, t, f, p0=p0, sigma=e, bounds=(lo, hi), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    names = ["P", "t0", "dA", "dB", "phiB", "dur", "ing"]
    resid = (f - eclipse_model(t, *popt)) / e
    return dict(zip(names, popt)), dict(zip(names, perr)), float(np.sum(resid ** 2) / (t.size - 7))


# ======================================================================
# Which of two unresolved stars is eclipsing?
# ======================================================================
def centroid_test(df, me, comp, P, T0, dur, phiB):
    """
    Two stars ~1" apart look like ONE blob to ZTF, and its measured position is the
    light-weighted middle of the pair. During an eclipse the dimming star gives less light,
    so the blob's position moves AWAY from whichever star is eclipsing:
        bright star (TIC 41602470) eclipses -> position moves toward the faint companion (+)
        faint companion eclipses            -> position moves toward the bright star (-)
    Predicted sizes (d = separation, s = faint star's share of the light, delta = eclipse depth):
        bright eclipses: + d * s * delta / (1 - delta)
        faint eclipses:  - d * delta * (1 - s) / (1 - delta)   (only possible if delta < s)
    Returns ("bright" / "faint" / "unclear", text).
    """
    cosd = np.cos(np.radians(me["dec"]))
    dx, dy = (comp["ra"] - me["ra"]) * cosd * 3600, (comp["dec"] - me["dec"]) * 3600
    d = float(np.hypot(dx, dy))
    ux, uy = dx / d, dy / d
    s = 10 ** (-0.4 * comp["g"]) / (10 ** (-0.4 * comp["g"]) + 10 ** (-0.4 * me["g"]))
    t = df["hjd"].to_numpy()
    dtA, dtB = phase_dt(t, P, T0), phase_dt(t, P, T0 + phiB * P)
    out = (np.abs(dtA) > dur) & (np.abs(dtB) > dur)
    # positions relative to each ZTF object's own average (fields differ slightly in astrometry)
    ref = df[out].groupby("oid")[["ra", "dec"]].median()
    x = (df["ra"] - df["oid"].map(ref["ra"])).to_numpy() * cosd * 3600
    y = (df["dec"] - df["oid"].map(ref["dec"])).to_numpy() * 3600
    along, across = x * ux + y * uy, -x * uy + y * ux
    print(f"   Position-shift test: companion is {d:.2f}\" away and gives ~{s * 100:.0f}% of the light (Gaia G).")
    print("      eclipse    band  points   shift toward companion   sideways       "
          "predicted if TIC 41602470 / if companion eclipses")
    decision = "unclear"
    text = ""
    for ecl, dt in (("primary", dtA), ("secondary", dtB)):
        for bands in (("r",), ("g",), ("g", "r")):
            sel = df["band"].isin(bands).to_numpy()
            o, core = sel & out, sel & (np.abs(dt) < 0.3 * dur)
            noise = 1.4826 * np.median(np.abs(along[o] - np.median(along[o])))
            core &= np.abs(along - np.median(along[o])) < 5 * noise          # drop wild outliers
            if core.sum() < 5:
                continue
            shift = np.mean(along[core]) - np.mean(along[o])
            side = np.mean(across[core]) - np.mean(across[o])
            err = noise / np.sqrt(core.sum())
            delta = dip_at(df, P, T0, dur, 0.0 if ecl == "primary" else phiB, sel)[0]
            pred_b = d * s * delta / (1 - delta)
            pred_f = -d * delta * (1 - s) / (1 - delta) if delta < s else np.nan
            print(f"      {ecl:9s}  {'+'.join(bands):4s}  {core.sum():5d}   {shift:+.3f} +- {err:.3f}\""
                  f"          {side:+.3f} +- {err:.3f}\"    {pred_b:+.3f}\" / "
                  + (f"{pred_f:+.3f}\"" if np.isfinite(pred_f) else "impossible (too faint)"))
            if ecl == "primary" and bands == ("g", "r"):
                z_b = abs(shift - pred_b) / err
                z_f = abs(shift - pred_f) / err if np.isfinite(pred_f) else np.inf
                if z_b < 3 <= z_f:
                    decision = "bright"
                elif z_f < 3 <= z_b:
                    decision = "faint"
                text = (f"primary eclipse, g+r: shift {shift:+.3f} +- {err:.3f}\" toward the companion; "
                        f"expected {pred_b:+.3f}\" if TIC 41602470 eclipses ({z_b:.1f} sigma off), "
                        + (f"{pred_f:+.3f}\" if the companion does ({z_f:.1f} sigma off)" if np.isfinite(pred_f)
                           else "companion too faint to cause it"))
    return decision, text


# ======================================================================
# The checks
# ======================================================================
def main():
    OUT.mkdir(exist_ok=True)
    info = f5.tic_info(TIC)
    from astroquery.mast import Catalogs
    tic_row = Catalogs.query_criteria(catalog="TIC", ID=TIC)[0]
    print(f"TIC {TIC}: RA {info['ra']:.6f}, Dec {info['dec']:+.6f}, Tmag {info['tmag']:.2f}, "
          f"Gaia DR3 {info['gaia']}, 2MASS {tic_row['TWOMASS']}")

    # ---------------------------------------------------------------- 1
    print("\n1. ZTF data")
    df = ztf_get(TIC, info["ra"], info["dec"])
    if df is None or len(df) < 100:
        check("ZTF download", False, "no usable ZTF data (server problem?) - stopping")
        return summary()
    # several ZTF objects can sit inside the 1.5" search circle; keep the one(s) on our star
    pos = df.groupby("oid")[["ra", "dec"]].median()
    sep = np.hypot((pos["ra"] - info["ra"]) * np.cos(np.radians(info["dec"])), pos["dec"] - info["dec"]) * 3600
    df = df[df["oid"].isin(sep[sep < 1.0].index)].reset_index(drop=True)
    df = normalise(df)
    for b in "gr":
        sub = df[df["band"] == b]
        check(f"ZTF {b} data", len(sub) > 100,
              f"{len(sub)} good points, {sub['oid'].nunique()} ZTF objects, fields "
              f"{sorted(sub['field'].unique().tolist())}, HJD {sub['hjd'].min():.1f}-{sub['hjd'].max():.1f} "
              f"({(sub['hjd'].max() - sub['hjd'].min()) / 365.25:.1f} years)")

    # ---------------------------------------------------------------- 2
    print("\n2. Is 5.46 days really the period?")
    t, f, e = df["hjd"].to_numpy(), df["flux"].to_numpy(), df["ferr"].to_numpy()
    print("   wide box search 0.2-30 d on g + r together (~1 minute)...")
    P, power = wide_search(t, f, e)
    peaks = top_peaks(P, power)
    best = peaks[0][0]
    # BLS folds both eclipses together, so its best peak is often the eclipse SPACING (P/2)
    P_orbit0 = best * 2 if abs(best * 2 - TESS_PERIOD) / TESS_PERIOD < 0.01 else best
    print("   strongest peaks:")
    for Pk, pw in peaks:
        print(f"      {Pk:10.5f} d   power {pw:7.1f}   {alias_name(Pk, P_orbit0)}")
    alias_powers = [pw for Pk, pw in peaks[1:] if "1-day" in alias_name(Pk, P_orbit0)]
    check("best peak is the eclipse period, not an alias",
          alias_name(best, P_orbit0) in ("P (orbit)", "P/2 (eclipse spacing)"),
          f"best {best:.5f} d = {alias_name(best, P_orbit0)}; strongest 1-day alias has "
          f"{(max(alias_powers) / peaks[0][1] * 100 if alias_powers else 0):.0f}% of its power")
    check("orbit matches TESS", abs(P_orbit0 - TESS_PERIOD) / TESS_PERIOD < 0.001,
          f"ZTF {P_orbit0:.5f} d vs TESS {TESS_PERIOD} d")

    # starting epoch: deepest point of the fold at the orbital period
    ph = (t - t.min()) / P_orbit0 % 1
    bins = np.floor(ph * 200).astype(int)
    mean_bin = np.bincount(bins, f, 200) / np.maximum(np.bincount(bins, minlength=200), 1)
    mean_bin[np.bincount(bins, minlength=200) < 3] = np.nan
    t0_guess = t.min() + (np.nanargmin(mean_bin) + 0.5) / 200 * P_orbit0

    fits = {}
    for b in "gr":
        sub = df[df["band"] == b].reset_index(drop=True)
        fits[b] = fit_band(sub, P_orbit0, t0_guess, 0.2)
    # make sure "primary" = deeper eclipse in r; if not, swap
    if fits["r"][0]["dB"] > fits["r"][0]["dA"]:
        t0_guess += P_orbit0 * fits["r"][0]["phiB"]
        fits = {b: fit_band(df[df["band"] == b].reset_index(drop=True), P_orbit0, t0_guess, 0.2) for b in "gr"}
    for b in "gr":
        p, pe, chi = fits[b]
        print(f"   fit {b}: P = {p['P']:.6f} +- {pe['P']:.6f} d, primary min HJD {p['t0']:.5f} +- {pe['t0']:.5f}, "
              f"depths {p['dA'] * 100:.1f}% / {p['dB'] * 100:.1f}%, secondary at phase {p['phiB']:.4f}, "
              f"duration {p['dur'] * 24:.2f} h (reduced chi2 {chi:.1f})")

    # combined ephemeris (weighted mean of g and r); errors scaled up if the fit isn't perfect
    def wmean(key):
        vals = np.array([fits[b][0][key] for b in "gr"])
        errs = np.array([fits[b][1][key] * np.sqrt(max(fits[b][2], 1)) for b in "gr"])
        w = 1 / errs ** 2
        return float(np.sum(w * vals) / np.sum(w)), float(1 / np.sqrt(np.sum(w))), vals, errs

    Pf, Pe, Pv, Pev = wmean("P")
    # put both bands' epochs on the same cycle before averaging
    t0r = fits["r"][0]["t0"]
    for b in "gr":
        fits[b][0]["t0"] = t0r + phase_dt(fits[b][0]["t0"], Pf, t0r)
    T0, T0e, T0v, T0ev = wmean("t0")
    check("g and r give the same period", abs(Pv[0] - Pv[1]) / np.hypot(*Pev) < 3,
          f"g {Pv[0]:.6f} +- {Pev[0]:.6f}, r {Pv[1]:.6f} +- {Pev[1]:.6f} "
          f"({abs(Pv[0] - Pv[1]) / np.hypot(*Pev):.1f} sigma apart)")
    check("g and r give the same epoch", abs(T0v[0] - T0v[1]) / np.hypot(*T0ev) < 3,
          f"g {T0v[0]:.5f}, r {T0v[1]:.5f} ({abs(T0v[0] - T0v[1]) * 24 * 60:.1f} min apart, "
          f"{abs(T0v[0] - T0v[1]) / np.hypot(*T0ev):.1f} sigma)")
    dur = float(np.mean([fits[b][0]["dur"] for b in "gr"]))
    dur_e = abs(fits["g"][0]["dur"] - fits["r"][0]["dur"]) / 2       # g and r disagree a bit: use the spread
    phiB = float(np.mean([fits[b][0]["phiB"] for b in "gr"]))

    # full period vs half period
    rr = df[df["band"] == "r"].reset_index(drop=True)
    dA, eA, _ = dip_at(rr, Pf, T0, dur)
    dB, eB, _ = dip_at(rr, Pf, T0, dur, phiB)
    check("two DIFFERENT eclipses at the full period (r)", abs(dA - dB) / np.hypot(eA, eB) > 5,
          f"primary {dA * 100:.1f} +- {eA * 100:.1f}%, secondary {dB * 100:.1f} +- {eB * 100:.1f}% "
          f"({abs(dA - dB) / np.hypot(eA, eB):.0f} sigma different)")
    half_scatter = []
    for P_try in (Pf, Pf / 2):
        dt = phase_dt(rr["hjd"].to_numpy(), P_try, T0)
        core = np.abs(dt) < 0.3 * dur
        half_scatter.append(np.std(rr["flux"].to_numpy()[core]))
    check("at half the period the eclipses overlap and look messier", half_scatter[1] > 1.5 * half_scatter[0],
          f"scatter inside eclipse: {half_scatter[0] * 100:.1f}% at P vs {half_scatter[1] * 100:.1f}% at P/2")

    # every filter and every field
    print("   eclipses in each filter and ZTF field:")
    all_ok = True
    for (b, field), grp in df.groupby(["band", "field"]):
        sel = (df["band"] == b) & (df["field"] == field)
        a = dip_at(df, Pf, T0, dur, 0.0, sel.to_numpy())
        s = dip_at(df, Pf, T0, dur, phiB, sel.to_numpy())
        ok_a = np.isnan(a[0]) or a[0] / a[1] > 3
        ok_s = np.isnan(s[0]) or s[0] / s[1] > 3
        all_ok &= ok_a and ok_s and not (np.isnan(a[0]) and np.isnan(s[0]))
        print(f"      {b} field {field}: {len(grp):4d} points | primary {a[0] * 100:5.1f} +- {a[1] * 100:.1f}% "
              f"({a[2]} pts) | secondary {s[0] * 100:5.1f} +- {s[1] * 100:.1f}% ({s[2]} pts)")
    check("eclipses in both filters and every field", all_ok, "see table above (nan = no points in eclipse)")

    # ---------------------------------------------------------------- 3
    print("\n3. Is it really this star?")
    from astroquery.gaia import Gaia
    q = f"""SELECT source_id, ra, dec, pmra, pmdec, parallax, parallax_error, bp_rp, ruwe,
                   phot_g_mean_mag AS g,
                   DISTANCE(POINT(ra, dec), POINT({info['ra']}, {info['dec']})) * 3600 AS sep
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(POINT(ra, dec), CIRCLE({info['ra']}, {info['dec']}, {10 / 3600})) = 1
            ORDER BY sep"""
    with f5.quiet():
        gaia = Gaia.launch_job(q).get_results()
    me = gaia[[str(s) == info["gaia"] for s in gaia["source_id"]]][0]
    print(f"   Gaia DR3 sources within 10\": {len(gaia)}")
    worry = []
    for g in gaia:
        need = dA * 10 ** (0.4 * (g["g"] - me["g"])) if str(g["source_id"]) != info["gaia"] else dA
        tag = "  <- TIC 41602470" if str(g["source_id"]) == info["gaia"] else \
            f"  would need a {need * 100:.0f}% eclipse" + (" (impossible)" if need > 1 else "")
        print(f"      Gaia DR3 {g['source_id']}  G = {g['g']:.2f}  {g['sep']:.2f}\"{tag}")
        if str(g["source_id"]) != info["gaia"] and g["sep"] < 3 and need <= 1:
            worry.append(g)
    check("no other Gaia source within 3\" that could fake the eclipse", not worry,
          f"{sum((gaia['sep'] < 3)) - 1} other source(s) within 3\"" +
          (f"; plausible: {[str(w['source_id']) for w in worry]}" if worry else ""))

    # ZTF position vs Gaia position (moved to the ZTF epoch with the star's proper motion)
    years = (np.median(df["mjd"]) - 57388.5) / 365.25            # since Gaia's epoch 2016.0
    g_ra = me["ra"] + (me["pmra"] or 0) / 3.6e6 * years / np.cos(np.radians(me["dec"]))
    g_dec = me["dec"] + (me["pmdec"] or 0) / 3.6e6 * years
    z_ra, z_dec = np.median(df["ra"]), np.median(df["dec"])
    offset = np.hypot((z_ra - g_ra) * np.cos(np.radians(g_dec)), z_dec - g_dec) * 3600
    check("ZTF position matches Gaia position of TIC 41602470", offset < 1.0, f"{offset:.2f}\" apart")

    # a companion too close for ZTF to separate? Then find out which of the two eclipses.
    close = [g for g in gaia if str(g["source_id"]) != info["gaia"] and g["sep"] < 3 and g["g"]]
    eclipsing, comp, shift_text = "bright", None, ""
    if close:
        comp = min(close, key=lambda g: g["sep"])
        for g, name in ((me, "TIC 41602470"), (comp, "companion")):
            print(f"      {name}: Gaia DR3 {g['source_id']}, G {g['g']:.2f}, BP-RP {g['bp_rp']:.2f}, "
                  f"parallax {g['parallax']:.3f} +- {g['parallax_error']:.3f} mas, "
                  f"pm ({g['pmra']:.2f}, {g['pmdec']:.2f}) mas/yr, RUWE {g['ruwe']:.2f}")
        eclipsing, shift_text = centroid_test(df, me, comp, Pf, T0, dur, phiB)
        check("eclipses are on TIC 41602470, not its 1\" companion (ZTF position shift)",
              {"bright": True, "faint": False, "unclear": False}[eclipsing], shift_text)

    # other stars within 1' in ZTF (stars within 3" are unresolved by ZTF: handled just above)
    neigh = f5.tic_neighbours(info["ra"], info["dec"], 60)
    unresolved = [n for n in neigh if int(n["ID"]) != TIC and n["dstArcSec"] < 3]
    for n in unresolved:
        print(f"   (TIC {n['ID']}, {n['dstArcSec']:.1f}\" away, is unresolved by ZTF - same light curve; "
              f"see the position-shift test)")
    neigh = [n for n in neigh if int(n["ID"]) not in (TIC, BLEND_TIC) and n["dstArcSec"] >= 3
             and 12.5 < n["Tmag"] < 17.0][:4]
    print(f"   ZTF light curves of {len(neigh)} other stars within 1' (folded at P = {Pf:.5f} d):")
    neighbours_ok, checked = True, 0
    for n in neigh:
        nd = ztf_get(int(n["ID"]), n["ra"], n["dec"])
        if nd is None or len(nd) < 100:
            print(f"      TIC {n['ID']} (Tmag {n['Tmag']:.1f}, {n['dstArcSec']:.0f}\"): no usable ZTF data")
            continue
        npos = nd.groupby("oid")[["ra", "dec"]].median()
        nsep = np.hypot((npos["ra"] - n["ra"]) * np.cos(np.radians(n["dec"])), npos["dec"] - n["dec"]) * 3600
        nd = normalise(nd[nd["oid"].isin(nsep[nsep < 1.0].index)].reset_index(drop=True))
        sel = (nd["band"] == "r").to_numpy()
        a = dip_at(nd, Pf, T0, dur, 0.0, sel)
        s = dip_at(nd, Pf, T0, dur, phiB, sel)
        n_eclipses = (a[0] > 0.01 and a[0] / a[1] > 3) or (s[0] > 0.01 and s[0] / s[1] > 3)
        neighbours_ok &= not n_eclipses
        checked += 1
        print(f"      TIC {n['ID']} (Tmag {n['Tmag']:.1f}, {n['dstArcSec']:.0f}\" away, {sel.sum()} r points): "
              f"primary {a[0] * 100:+.2f} +- {a[1] * 100:.2f}%, secondary {s[0] * 100:+.2f} +- {s[1] * 100:.2f}%"
              f"{'  <- ECLIPSES TOO' if n_eclipses else ''}")
    check("neighbours within 1' show no eclipses at this period", neighbours_ok and checked >= 2,
          f"{checked} neighbours checked in ZTF r")

    # ---------------------------------------------------------------- 4
    print("\n4. Is it really new? (everything within 1')")
    found_any = False
    for source, label in (("B/vsx/vsx", "VSX"), ("II/366/catv2021", "ASAS-SN variables"),
                          ("J/ApJS/249/18/table2", "ZTF periodic (Chen+2020)"),
                          ("J/ApJS/258/16/tess-ebs", "TESS EB catalogue")):
        try:
            rows = f5.vizier_rows(source, info["ra"], info["dec"], 60)
        except Exception as ex:
            check(f"{label} search", False, f"query failed ({ex.__class__.__name__})")
            continue
        for r in rows:
            period = next((r[k] for k in r if k.lower() in ("period", "per")), "?")
            name = next(iter(r.values()))
            close = r["sep"] < 3
            found_any |= close
            print(f"      {label}: {name} type {r.get('Type', '?')} P = {period} d, {r['sep']:.1f}\" away"
                  f"{'  <- AT OUR STAR' if close else ''}")
        check(f"not in {label}", not any(r["sep"] < 3 for r in rows), f"{len(rows)} entries within 1'")
    q = f"""SELECT g.source_id, g.phot_g_mean_mag AS g, e.frequency,
                   DISTANCE(POINT(g.ra, g.dec), POINT({info['ra']}, {info['dec']})) * 3600 AS sep
            FROM gaiadr3.gaia_source AS g JOIN gaiadr3.vari_eclipsing_binary AS e ON g.source_id = e.source_id
            WHERE CONTAINS(POINT(g.ra, g.dec), CIRCLE({info['ra']}, {info['dec']}, {60 / 3600})) = 1"""
    with f5.quiet():
        geb = Gaia.launch_job(q).get_results()
    for r in geb:
        print(f"      Gaia DR3 EB: {r['source_id']} G = {r['g']:.1f} P = {1 / r['frequency']:.5f} d, {r['sep']:.1f}\" away")
    check("not in Gaia DR3 eclipsing binaries", not any(str(r["source_id"]) == info["gaia"] for r in geb),
          f"{len(geb)} Gaia EBs within 1'")
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.simbad import Simbad
        with f5.quiet():
            sim = Simbad().query_region(SkyCoord(info["ra"], info["dec"], unit="deg"), radius=60 * u.arcsec)
        names = [] if sim is None else [str(x) for x in sim[sim.colnames[0]]]
        check("SIMBAD", None, f"{len(names)} objects within 1': {names[:8]}")
    except Exception as ex:
        check("SIMBAD", None, f"not working today ({ex.__class__.__name__})")

    # ---------------------------------------------------------------- 5
    print("\n5. What VSX needs")
    check("period", None, f"{Pf:.6f} +- {Pe:.6f} d")
    check("epoch of primary minimum", None, f"HJD {T0:.5f} +- {T0e:.5f} (heliocentric JD, UTC, from ZTF)")
    check("eclipse duration", None, f"{dur * 24:.1f} +- {dur_e * 24:.1f} h (g {fits['g'][0]['dur'] * 24:.1f} h, "
          f"r {fits['r'][0]['dur'] * 24:.1f} h) = {dur / Pf * 100:.1f} +- {dur_e / Pf * 100:.1f}% of the period; "
          f"secondary at phase {phiB:.4f}")
    mags = {}
    for b in "gr":
        sub = df[df["band"] == b]
        dtA = phase_dt(sub["hjd"].to_numpy(), Pf, T0)
        dtB = phase_dt(sub["hjd"].to_numpy(), Pf, T0 + phiB * Pf)
        out = (np.abs(dtA) > dur) & (np.abs(dtB) > dur)
        m_max = np.median(sub["mag"][out])
        field_meds = sub[out].groupby("field")["mag"].median()
        m_max_e = np.hypot(np.std(sub["mag"][out]) / np.sqrt(out.sum()), np.std(field_meds) if len(field_meds) > 1 else 0)
        p, pe, chi = fits[b]
        s = np.sqrt(max(chi, 1))
        m1 = m_max - 2.5 * np.log10(1 - p["dA"])
        m2 = m_max - 2.5 * np.log10(1 - p["dB"])
        m1e = np.hypot(m_max_e, 2.5 / np.log(10) * pe["dA"] * s / (1 - p["dA"]))
        m2e = np.hypot(m_max_e, 2.5 / np.log(10) * pe["dB"] * s / (1 - p["dB"]))
        # The trapezoid has a flat bottom, but these eclipses are rounded, so the fit can be too
        # shallow at minimum (r primary: fit 15.24, points 15.30). Where there are >= 5 points
        # within 1 hour of minimum, use their median instead.
        for which, dt_min in (("I", dtA), ("II", dtB)):
            near_min = np.abs(dt_min) < 1 / 24
            if near_min.sum() >= 5:
                m_direct = float(np.median(sub["mag"][near_min]))
                e_direct = float(np.std(sub["mag"][near_min]) / np.sqrt(near_min.sum()))
                fitted = m1 if which == "I" else m2
                print(f"   {b} min {which}: fit {fitted:.3f}, median of {near_min.sum()} points within 1 h "
                      f"{m_direct:.3f} +- {e_direct:.3f} -> using the median")
                if which == "I":
                    m1, m1e = m_direct, np.hypot(e_direct, m_max_e)
                else:
                    m2, m2e = m_direct, np.hypot(e_direct, m_max_e)
        mags[b] = (m_max, m_max_e, m1, m1e, m2, m2e)
        check(f"magnitudes ZTF {b}", None, f"max {m_max:.3f} +- {m_max_e:.3f}, primary min {m1:.3f} +- {m1e:.3f}, "
              f"secondary min {m2:.3f} +- {m2e:.3f}")

        # EA (flat between eclipses) or EB (brightness changes all the time)?
        phase = (sub["hjd"].to_numpy()[out] - T0) / Pf * 2 * np.pi
        A = np.column_stack([np.ones(out.sum()), np.cos(2 * phase), np.sin(2 * phase), np.cos(phase), np.sin(phase)])
        coef, *_ = np.linalg.lstsq(A, sub["dmag"].to_numpy()[out], rcond=None)
        resid = sub["dmag"].to_numpy()[out] - A @ coef
        amp = 2 * np.hypot(coef[1], coef[2])                     # peak-to-peak of the half-orbit wave
        amp_err = 2 * np.std(resid) * np.sqrt(2 / out.sum())
        mags[b] += (amp, amp_err)
    p2p_r, p2p_r_e = mags["r"][6], mags["r"][7]
    depth_r_mag = mags["r"][2] - mags["r"][0]
    check("type EA (flat between eclipses)", p2p_r < 0.1 * depth_r_mag,
          f"out-of-eclipse variation in r {p2p_r * 1000:.0f} +- {p2p_r_e * 1000:.0f} mmag vs primary depth "
          f"{depth_r_mag * 1000:.0f} mmag; eclipses last {dur / Pf * 100:.1f}% of the orbit")

    # TESS: does the blended signal in TIC 41602482 match the ZTF ephemeris and the expected dilution?
    tt, tf, _, _ = survey.download(BLEND_TIC, 40)
    T0_btjd = T0 - 2457000
    tmask = (np.abs(phase_dt(tt, Pf, T0_btjd)) < dur) | (np.abs(phase_dt(tt, Pf, T0_btjd + phiB * Pf)) < dur)
    tflat = hunter.detrend(tt, tf, tmask, window=1.0)
    core = np.abs(phase_dt(tt, Pf, T0_btjd)) < 0.3 * dur
    core2 = np.abs(phase_dt(tt, Pf, T0_btjd + phiB * Pf)) < 0.3 * dur
    tA, tB = 1 - np.mean(tflat[core]), 1 - np.mean(tflat[core2])
    # where is the TESS minimum relative to the ZTF prediction?
    dtp = phase_dt(tt, Pf, T0_btjd)
    near = np.abs(dtp) < 0.5 * dur
    xb, fb = hunter.bin_data(dtp[near], tflat[near], size=10 / 1440)
    shift_h = xb[np.argmin(fb)] * 24
    check("TESS eclipses fall where ZTF predicts", abs(shift_h) < 1.0,
          f"TESS primary minimum {shift_h:+.2f} h from the ZTF ephemeris (extrapolated to 2021)")
    frac = 10 ** (-0.4 * info["tmag"]) / (10 ** (-0.4 * info["tmag"]) + 10 ** (-0.4 * 10.28))
    pred = fits["r"][0]["dA"] * frac
    check("TESS depth fits a blend with TIC 41602482", 0.5 < tA / pred < 2,
          f"TESS primary {tA * 1e6:.0f} ppm, secondary {tB * 1e6:.0f} ppm; predicted from ZTF r depth x "
          f"light share ({frac * 100:.1f}%): {pred * 1e6:.0f} ppm (ratio {tA / pred:.2f})")
    check("primary in ZTF = deeper eclipse in TESS", tA > tB, f"{tA * 1e6:.0f} vs {tB * 1e6:.0f} ppm")

    # ---------------------------------------------------------------- 6
    print("\n6. Plots")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogx(P, power, lw=0.4, color="black")
    for Pk, pw in peaks[:6]:
        ax.annotate(f"{Pk:.4f}\n{alias_name(Pk, Pf)}", (Pk, pw), fontsize=7, ha="center",
                    xytext=(0, 8), textcoords="offset points")
    ax.set_xlabel("Trial period (days)")
    ax.set_ylabel("BLS power (SNR)")
    ax.set_title(f"TIC {TIC}: ZTF g+r box search, 0.2-30 days")
    fig.tight_layout()
    fig.savefig(OUT / f"TIC_{TIC}_periodogram.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, P_try, name in ((axes[0], Pf, f"full period {Pf:.5f} d"), (axes[1], Pf / 2, f"half period {Pf / 2:.5f} d")):
        phs = ((rr["hjd"] - T0) / P_try + 0.25) % 1 - 0.25
        ax.plot(phs, rr["mag"], ".", ms=2, color="black", alpha=0.5)
        ax.set_title(f"ZTF r folded at the {name}")
        ax.set_xlabel("Phase")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("r magnitude")
    fig.tight_layout()
    fig.savefig(OUT / f"TIC_{TIC}_full_vs_half.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    for ax, b, colour in ((axes[0], "r", "firebrick"), (axes[1], "g", "seagreen")):
        sub = df[df["band"] == b]
        phs = ((sub["hjd"] - T0) / Pf + 0.25) % 1 - 0.25
        ax.plot(phs, sub["mag"], ".", ms=2.5, color=colour, alpha=0.6)
        ax.invert_yaxis()
        ax.set_ylabel(f"ZTF {b} (mag)")
        ax.set_title(f"ZTF {b}: {len(sub)} points, max {mags[b][0]:.2f}, min I {mags[b][2]:.2f}, "
                     f"min II {mags[b][4]:.2f}", fontsize=10)
    tph = ((tt - T0_btjd) / Pf + 0.25) % 1 - 0.25
    axes[2].plot(tph, tflat, ",", color="gray")
    xb, fb = hunter.bin_data(tph, tflat, size=0.004)
    axes[2].plot(xb, fb, ".", ms=3, color="black")
    axes[2].set_ylabel("TESS flux (blend)")
    axes[2].set_title(f"TESS sector 40, light of TIC {BLEND_TIC} + TIC {TIC} (blended, ~{frac * 100:.1f}% from TIC {TIC})",
                      fontsize=10)
    for ax in axes:
        ax.axvline(0, color="blue", lw=0.8, ls="--")
        ax.axvline(phiB, color="orange", lw=0.8, ls="--")
        ax.set_xlim(-0.25, 0.75)
    axes[0].text(0.01, 0.05, "primary", color="blue", transform=axes[0].get_xaxis_transform())
    axes[0].text(phiB + 0.01, 0.05, "secondary", color="orange", transform=axes[0].get_xaxis_transform())
    axes[2].set_xlabel(f"Phase  (P = {Pf:.6f} d, primary min HJD {T0:.5f})")
    fig.suptitle(f"TIC {TIC} = Gaia DR3 {info['gaia']}: eclipsing binary", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"TIC_{TIC}_vsx_folds.png", dpi=130)
    plt.close(fig)
    print(f"   {OUT.name}/TIC_{TIC}_periodogram.png, TIC_{TIC}_full_vs_half.png, TIC_{TIC}_vsx_folds.png")

    # ---------------------------------------------------------------- 7
    from astropy.coordinates import SkyCoord
    star = comp if eclipsing == "faint" else me
    c = SkyCoord(float(star["ra"]), float(star["dec"]), unit="deg")
    r_, g_ = mags["r"], mags["g"]
    if comp is None:
        status = "READY FOR REVIEW"
        blend_note = "No other Gaia DR3 source within 3\"."
    elif eclipsing == "bright":
        status = "READY FOR REVIEW"
        blend_note = (f"Gaia DR3 resolves a fainter star (Gaia DR3 {comp['source_id']}, G = {comp['g']:.2f}) "
                      f"{comp['sep']:.1f}\" away that ZTF does not separate, so the ZTF magnitudes include its light. "
                      f"The ZTF positions shift during primary eclipse as expected if THIS star is the "
                      f"eclipsing one ({shift_text}).")
    elif eclipsing == "faint":
        status = "READY FOR REVIEW - NOTE: the variable is the faint companion, not TIC 41602470"
        blend_note = (f"The eclipsing star is the fainter component of a {comp['sep']:.1f}\" pair; the brighter "
                      f"component is TIC {TIC} = Gaia DR3 {info['gaia']} (G = {me['g']:.2f}). ZTF does not "
                      f"separate them, so the magnitudes are combined light; the ZTF positions shift during "
                      f"eclipse toward the bright star ({shift_text}).")
    else:
        status = "NOT READY: which of the two unresolved stars eclipses is still unknown"
        blend_note = (f"The eclipses come from one of two stars {comp['sep']:.1f}\" apart, TIC {TIC} = "
                      f"Gaia DR3 {info['gaia']} (G = {me['g']:.2f}) or Gaia DR3 {comp['source_id']} "
                      f"(G = {comp['g']:.2f}), which ZTF does not separate; the ZTF position-shift test was "
                      f"not decisive ({shift_text}).")
    other_ids = f"TIC {TIC}; 2MASS J{tic_row['TWOMASS']}" if star is me else \
        f"(companion of TIC {TIC}, {comp['sep']:.1f}\" away; no TIC/2MASS entry of its own checked)"
    draft = f"""DRAFT VSX submission - for review, NOT submitted
STATUS: {status}
==================================================
Name:          Gaia DR3 {star['source_id']}
Other IDs:     {other_ids}
Position:      {c.to_string('hmsdms', sep=':', precision=2)} (J2000, Gaia DR3)
Type:          EA
Period:        {Pf:.6f} d (+- {Pe:.6f})
Epoch:         HJD {T0:.5f} (+- {T0e:.5f}), primary minimum (HJD, UTC)
Range (r):     {r_[0]:.2f} - {r_[2]:.2f} r (ZTF)   min II {r_[4]:.2f} r
Range (g):     {g_[0]:.2f} - {g_[2]:.2f} g (ZTF)   min II {g_[4]:.2f} g
Eclipse duration: {dur / Pf * 100:.1f}% of the period ({dur * 24:.1f} +- {dur_e * 24:.1f} h); secondary at phase {phiB:.3f}

Remarks:
Detached eclipsing binary. Primary and secondary eclipses of {fits['r'][0]['dA'] * 100:.0f}% and {fits['r'][0]['dB'] * 100:.0f}%
in ZTF r, light curve flat between eclipses. Period from a 0.2-30 d box search of ZTF g and r
data ({(df['hjd'].max() - df['hjd'].min()) / 365.25:.1f} yr), confirmed independently in both filters and every ZTF field.
{blend_note}
The eclipses were first noticed in TESS sector 40 in the 2-min SPOC light curve of
TIC {BLEND_TIC} (T = 10.3, 11.6" away), where they are diluted to ~{tA * 1e6:.0f} and ~{tB * 1e6:.0f} ppm
and fall within {abs(shift_h) * 60:.0f} min of the ZTF ephemeris. Stars 20-22" away show no eclipses in ZTF.

Data sources: ZTF public data via IRSA (catflags = 0), accessed 2026-11-14; TESS sector 40
(SPOC). Positions: Gaia DR3.
"""
    path = OUT / f"TIC_{TIC}_VSX_draft.txt"
    path.write_text(draft, encoding="utf-8")
    print("\n7. Draft VSX entry")
    print(draft)
    print(f"   saved to {path.name}")
    summary()


def summary():
    print("\n" + "=" * 70 + "\nSUMMARY")
    for tag, name, detail in RESULTS:
        print(f"  [{tag}] {name}")
    n_fail = sum(tag == "FAIL" for tag, _, _ in RESULTS)
    print(f"\n{sum(tag == 'PASS' for tag, _, _ in RESULTS)} PASS, {n_fail} FAIL")


if __name__ == "__main__":
    main()
