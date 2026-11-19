"""
Planet hunt, step 5: follow-up tests on the interesting stars from step 4.

TIC 41602482: repeating V-shaped dips, not in VSX / TESS EB / Kepler EB catalogues.
    1. Fold with step 2 and compare odd and even dips (do the depths alternate?)
    2. Look up VSX and Gaia DR3's eclipsing-binary list (~2 million) at its position
    3. Pixel test: which star in the image actually gets dimmer?

TIC 356602976: one lone ~14-hour dip at day 2414.35 in sector 40.
    1. Search every OTHER TESS sector of this star for a second dip
    2. Raw (SAP) flux, background light and quality flags around the dip
    3. Do other sector-40 stars dip at the same moment? (that would be the telescope)
    4. Pixel test

The pixel test, in plain words
    A TESS pixel is 21 arcseconds wide, so several stars can share pixels. We make
    two pictures of the star: the average image OUTSIDE the dip, and the average image
    DURING the dip. Subtracting them ("difference image") shows which pixels lost
    light. If the loss is centred on our star, it's our star dimming. If it's centred
    somewhere else, a neighbour is the one dimming.

Run:   python step5_followup.py
Graphs go in the "followup" folder.
"""
import contextlib
import io
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import step2_hunt as hunter
import step3_survey as survey
import step4_single_transits as single

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "followup"
SECTOR = 40
PIXEL_ARCSEC = 21.0


def quiet():
    """Hide the download library's chatter."""
    return contextlib.redirect_stdout(io.StringIO())


def run_test(name, func, *args):
    """Run one test; if it crashes, show why and carry on with the next one."""
    print(f"\n--- {name} ---")
    try:
        return func(*args)
    except Exception:
        traceback.print_exc()


# ======================================================================
# Catalogue helpers
# ======================================================================
def tic_info(tic):
    from astroquery.mast import Catalogs
    row = Catalogs.query_criteria(catalog="TIC", ID=tic)[0]
    return {"ra": float(row["ra"]), "dec": float(row["dec"]), "tmag": float(row["Tmag"]),
            "gaia": str(row["GAIA"])}


def tic_neighbours(ra, dec, radius_arcsec=120):
    """All TIC stars within radius, nearest first (includes the star itself)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs
    tab = Catalogs.query_region(SkyCoord(ra, dec, unit="deg"), radius=radius_arcsec * u.arcsec,
                                catalog="TIC")
    tab = tab[~tab["Tmag"].mask] if hasattr(tab["Tmag"], "mask") else tab
    tab.sort("dstArcSec")
    return tab


def vizier_rows(source, ra, dec, radius_arcsec):
    """Rows of one VizieR catalogue near a position, with distance 'sep' in arcsec."""
    import requests
    text = requests.get(single.VIZIER, timeout=60, params={
        "-source": source, "-c": f"{ra} {dec:+}", "-c.rs": radius_arcsec,
        "-out.add": "_r", "-sort": "_r", "-out.max": 50}).text
    rows, header, units, in_data = [], None, None, False
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            in_data = False
        elif header is None:
            header = [h.strip() for h in line.split("\t")]
        elif units is None:
            units = dict(zip(header, (u.strip() for u in line.split("\t"))))
        elif line.startswith("-"):
            in_data = True
        elif in_data:
            row = dict(zip(header, (v.strip() for v in line.split("\t"))))
            scale = {"deg": 3600, "arcmin": 60, "arcsec": 1}.get(units.get("_r", "arcmin"), 60)
            row["sep"] = float(row["_r"]) * scale
            rows.append(row)
    return rows


def catalogue_check(info, radius_arcsec=60):
    """VSX and Gaia DR3 eclipsing binaries at (and around) the star's position."""
    ra, dec = info["ra"], info["dec"]
    print(f"Position RA {ra:.5f}, Dec {dec:+.5f}; Gaia DR3 {info['gaia']}; searching {radius_arcsec}\"")

    vsx = vizier_rows("B/vsx/vsx", ra, dec, radius_arcsec)
    print(f"VSX: {len(vsx)} variable star(s) within {radius_arcsec}\"")
    for r in vsx:
        print(f"   {r.get('Name', '?'):28s} type {r.get('Type', '?'):8s} P={r.get('Period', '') or '?'} d"
              f"   {r['sep']:.1f}\" away")

    from astroquery.gaia import Gaia
    query = f"""
        SELECT g.source_id, g.phot_g_mean_mag AS gmag,
               DISTANCE(POINT(g.ra, g.dec), POINT({ra}, {dec})) * 3600 AS sep,
               e.frequency, e.global_ranking, e.model_type
        FROM gaiadr3.gaia_source AS g
        JOIN gaiadr3.vari_eclipsing_binary AS e ON g.source_id = e.source_id
        WHERE CONTAINS(POINT(g.ra, g.dec), CIRCLE({ra}, {dec}, {radius_arcsec / 3600})) = 1
        ORDER BY sep"""
    with quiet():
        gaia = Gaia.launch_job(query).get_results()
    print(f"Gaia DR3 eclipsing binaries: {len(gaia)} within {radius_arcsec}\"")
    for r in gaia:
        period = 1 / r["frequency"] if r["frequency"] else float("nan")
        me = "  <- THIS STAR" if str(r["source_id"]) == info["gaia"] else ""
        print(f"   Gaia DR3 {r['source_id']}  G={r['gmag']:.1f}  P={period:.4f} d  "
              f"{r['sep']:.1f}\" away{me}")
    return vsx, gaia


# ======================================================================
# Pixel test
# ======================================================================
def pixel_test(tic, info, in_event, near_event, depth, title, out_png):
    """
    in_event(t), near_event(t): which moments are inside the dip, and which are
    just outside it (the comparison). Prints where the lost light is centred.
    """
    import lightkurve as lk
    from astropy.coordinates import SkyCoord
    with quiet():
        tpf = lk.search_targetpixelfile(f"TIC {tic}", mission="TESS", author="SPOC",
                                        exptime=120, sector=SECTOR)[0].download()
    t = tpf.time.value
    flux = tpf.flux.value                               # (time, row, column)
    good = np.isfinite(t) & (tpf.quality == 0)
    inn, near = good & in_event(t), good & near_event(t)
    print(f"Pixel cutout {flux.shape[2]} x {flux.shape[1]}; {inn.sum()} images in the dip, "
          f"{near.sum()} just outside")

    img_out = np.nanmean(flux[near], axis=0)
    img_in = np.nanmean(flux[inn], axis=0)
    diff = img_out - img_in                             # light LOST during the dip
    # Remove a smooth background slope across the cutout first (scattered light changing during
    # the dip). Otherwise the slope, not a star, decides where the "lost light" seems to be
    # (TIC 138131958, 198099774). Fitted to the fainter half of the pixels (mostly background).
    rows, cols = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    bg = np.isfinite(diff) & (img_out < np.nanpercentile(img_out, 50))
    if bg.sum() >= 10:
        A = np.column_stack([np.ones(bg.sum()), cols[bg], rows[bg]])
        coef, *_ = np.linalg.lstsq(A, diff[bg], rcond=None)
        diff = diff - (coef[0] + coef[1] * cols + coef[2] * rows)
    err = np.sqrt(np.nanvar(flux[near], axis=0) / near.sum() + np.nanvar(flux[inn], axis=0) / inn.sum())
    diff_snr = diff / err

    # where is the lost light centred? (3x3 pixels around the strongest pixel)
    r0, c0 = np.unravel_index(np.nanargmax(np.where(np.isfinite(diff_snr), diff_snr, -np.inf)), diff.shape)
    rows, cols = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    box = (np.abs(rows - r0) <= 1) & (np.abs(cols - c0) <= 1)
    w = np.clip(np.nan_to_num(diff), 0, None) * box
    cen_x, cen_y = (w * cols).sum() / w.sum(), (w * rows).sum() / w.sum()

    # where the stars are, in pixel coordinates
    neigh = tic_neighbours(info["ra"], info["dec"], 150)
    nx, ny = tpf.wcs.world_to_pixel(SkyCoord(neigh["ra"], neigh["dec"], unit="deg"))
    tx, ty = tpf.wcs.world_to_pixel(SkyCoord(info["ra"], info["dec"], unit="deg"))
    offset = np.hypot(cen_x - tx, cen_y - ty) * PIXEL_ARCSEC

    print(f"Strongest light-loss pixel: SNR {diff_snr[r0, c0]:.1f}"
          + ("  -> too weak to trust (noise), so the centre below is NOT reliable"
             if diff_snr[r0, c0] < 5 else ""))
    print(f"Lost light is centred {offset:.1f}\" ({offset / PIXEL_ARCSEC:.2f} pixels) from TIC {tic}")

    def snr_at(x, y):
        c, r = int(round(x)), int(round(y))
        if 0 <= r < diff.shape[0] and 0 <= c < diff.shape[1]:
            return f"{diff_snr[r, c]:5.1f}"
        return "  off"

    print("Nearby stars (could one of them be the real source?):")
    print("   TIC            Tmag   from target   from light-loss centre   light-loss SNR at its pixel"
          "   dip it would need")
    for i in range(len(neigh)):
        if neigh["Tmag"][i] > info["tmag"] + 7 or neigh["dstArcSec"][i] > 100:
            continue
        need = depth * 10 ** (0.4 * (neigh["Tmag"][i] - info["tmag"]))
        d_cen = np.hypot(nx[i] - cen_x, ny[i] - cen_y) * PIXEL_ARCSEC
        need_txt = f"{need * 100:.2f}%" + ("  (impossible, >100%)" if need > 1 else "")
        print(f"   {neigh['ID'][i]:<13s} {neigh['Tmag'][i]:5.1f}   {neigh['dstArcSec'][i]:6.1f}\""
              f"        {d_cen:6.1f}\"                {snr_at(nx[i], ny[i])}"
              f"                     {need_txt}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [(img_out, "Average image outside the dip", "viridis"),
              (diff, "Difference: light lost during the dip", "RdBu_r"),
              (diff_snr, "Difference / noise (SNR)", "RdBu_r")]
    for ax, (img, name, cmap) in zip(axes, panels):
        if cmap == "RdBu_r":
            lim = np.nanmax(np.abs(img))
            im = ax.imshow(img, origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(np.log10(np.clip(img, 1, None)), origin="lower", cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)
        inside = (nx > -0.5) & (nx < img.shape[1] - 0.5) & (ny > -0.5) & (ny < img.shape[0] - 0.5)
        bright = neigh["Tmag"] < info["tmag"] + 5
        ax.scatter(nx[inside & bright], ny[inside & bright], s=40, facecolors="none",
                   edgecolors="white" if cmap == "viridis" else "black", label="other stars (Tmag < target+5)")
        ax.plot(tx, ty, "x", color="red", ms=12, mew=2, label=f"TIC {tic}")
        ax.plot(cen_x, cen_y, "+", color="orange", ms=14, mew=2, label="light-loss centre")
        ax.set_title(name, fontsize=10)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Graph: {out_png.name}")
    return offset


# ======================================================================
# TIC 41602482: the suspected eclipsing binary
# ======================================================================
def fold_binary(tic):
    """Step 2's search and odd/even test; then show each dip's depth in order."""
    t, f, tmag, radius = survey.download(tic, SECTOR)
    print(f"{t.size} points; star radius {radius:.2f} Suns (big: an evolved star)")
    # Step 2's BLS search, aimed at the dip spacing we can see (~2.7 d). (Step 2's full
    # hunt() found nothing past its strict thresholds on this wobbly giant star.)
    flat = hunter.detrend(t, f, np.zeros(t.size, bool), window=1.0)
    tb, fb = hunter.bin_data(t, flat)
    _, cand = hunter.bls_search(tb, fb, np.linspace(2.4, 3.1, 3000))
    P, t0, D = cand["period"], cand["t0"], cand["duration"]
    # clean again, protecting the dips, then pin the period down
    flat = hunter.detrend(t, f, hunter.in_transit(t, P, t0, 1.5 * D), window=1.0)
    tb, fb = hunter.bin_data(t, flat)
    cand = hunter.refine(tb, fb, cand)
    P, t0, D = cand["period"], cand["t0"], cand["duration"]
    print(f"Step 2 finds dips every {P:.4f} d (so the full orbit would be {2 * P:.4f} d), "
          f"each ~{D * 24:.1f} h long")

    v = hunter.vet(t, flat, {"period": P, "t0": t0, "duration": D})
    print(f"Odd dips:  {v['odd'] * 1e6:6.0f} ppm deep")
    print(f"Even dips: {v['even'] * 1e6:6.0f} ppm deep")
    print(f"Difference: {v['odd_even_sigma']:.1f} sigma "
          f"({'they ALTERNATE -> two different stars eclipsing each other' if v['odd_even_sigma'] > 3 else 'not clearly different'})")

    # every dip, one by one
    noise = np.std(flat[~hunter.in_transit(t, P, t0, 2 * D)])
    print("Each dip in order (depth of the central part):")
    for n in range(int(np.floor((t.min() - t0) / P)), int(np.ceil((t.max() - t0) / P)) + 1):
        sel = np.abs(t - (t0 + n * P)) < 0.3 * D
        if sel.sum() > 20:
            d = 1 - np.mean(flat[sel])
            print(f"   day {t0 + n * P:8.2f}  {'A' if n % 2 == 0 else 'B'}  {d * 1e6:5.0f} ppm"
                  f"  (+- {noise / np.sqrt(sel.sum()) * 1e6:.0f})")

    # fold at the full orbit: eclipse A and eclipse B side by side
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    phase = ((t - t0 + 0.25 * 2 * P) % (2 * P)) - 0.25 * 2 * P        # days from nearest A, range -P/2..1.5P
    for ax, centre, name in ((axes[0], 0.0, "eclipse A (even)"), (axes[1], P, "eclipse B (odd)")):
        x = (phase - centre) * 24
        sel = np.abs(x) < 3 * D * 24
        ax.plot(x[sel], flat[sel], ".", ms=1, color="lightgray")
        xb, fb = hunter.bin_data(x[sel], flat[sel], size=0.25)
        ax.plot(xb, fb, "o", ms=3, color="red")
        ax.set_title(name)
        ax.set_xlabel("Hours from eclipse centre")
    axes[0].set_ylabel("Brightness")
    for ax, d in zip(axes, (v["even"], v["odd"])):
        ax.axhline(1 - d, color="red", lw=0.8, ls="--")
    fig.suptitle(f"TIC {tic} folded at {2 * P:.4f} d: even {v['even'] * 1e6:.0f} ppm vs "
                 f"odd {v['odd'] * 1e6:.0f} ppm ({v['odd_even_sigma']:.1f} sigma)")
    fig.tight_layout()
    out = OUT / f"TIC_{tic}_fold.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Graph: {out.name}")
    return P, t0, D, max(v["odd"], v["even"])


def ztf_lightcurve(ra, dec, radius_arcsec=2.0):
    """
    ZTF (Zwicky Transient Facility) light curve from NASA's IRSA archive. ZTF has 1"
    pixels, so unlike TESS it sees each neighbour separately. Returns time (TESS days),
    normalised flux, or None if ZTF has nothing (stars brighter than ~mag 12.5 saturate).
    """
    import pandas as pd
    import requests
    text = requests.get("https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves", timeout=180, params={
        "POS": f"CIRCLE {ra} {dec} {radius_arcsec / 3600}", "BANDNAME": "g,r",
        "FORMAT": "csv", "BAD_CATFLAGS_MASK": 32768}).text
    df = pd.read_csv(io.StringIO(text))
    df = df[df["catflags"] == 0] if "catflags" in df else df
    if len(df) < 50:
        return None
    # each ZTF "object" (field + camera chip + filter) has its own brightness zero point
    df["flux"] = 10 ** (-0.4 * (df["mag"] - df.groupby("oid")["mag"].transform("median")))
    return df["hjd"].to_numpy() - 2457000, df["flux"].to_numpy(), df


def ztf_check(tic=41602482, P=5.4574, t0=2391.16, D=0.2, radius_arcsec=30):
    """Which star near TIC 41602482 shows eclipses at the TESS period in ZTF?"""
    info = tic_info(tic)
    neigh = tic_neighbours(info["ra"], info["dec"], radius_arcsec)
    neigh = neigh[neigh["Tmag"] < 17.5]
    print(f"Fold period {P} d (from TESS); eclipse A at day {t0}, B half an orbit later")
    print("   TIC            Tmag   away    ZTF points   eclipse A     eclipse B     best ZTF period (BLS)")
    best_star = None
    for row in neigh:
        name = f"{row['ID']:<13s} {row['Tmag']:5.1f}  {row['dstArcSec']:5.1f}\""
        try:
            lc = ztf_lightcurve(row["ra"], row["dec"])
        except Exception as e:
            print(f"   {name}   ZTF query failed ({e.__class__.__name__})")
            continue
        if lc is None:
            print(f"   {name}   no usable ZTF data" + ("  (too bright: saturated)" if row["Tmag"] < 12.5 else ""))
            continue
        t, f, df = lc
        phase = ((t - t0) / P + 0.25) % 1 - 0.25          # 0 = eclipse A, 0.5 = eclipse B
        w = 0.4 * D / P
        out = (np.abs(phase) > 2.5 * w) & (np.abs(phase - 0.5) > 2.5 * w)
        noise = np.std(f[out])

        def dip(sel):
            return (1 - np.mean(f[sel]), noise / np.sqrt(sel.sum())) if sel.sum() >= 3 else (np.nan, np.nan)

        dA, eA = dip(np.abs(phase) < w)
        dB, eB = dip(np.abs(phase - 0.5) < w)
        _, cand = hunter.bls_search(t, f, np.linspace(5.40, 5.52, 6000))
        print(f"   {name}   {t.size:6d}      {dA * 100:5.1f}%+-{eA * 100:.1f}  {dB * 100:5.1f}%+-{eB * 100:.1f}"
              f"   {cand['period']:.4f} d (SNR {cand['snr']:.0f})")
        score = np.nansum([dA / eA, dB / eB])
        if best_star is None or score > best_star[0]:
            best_star = (score, row, t, f, cand)

    if best_star is None:
        return
    score, row, t, f, cand = best_star
    print(f"\nStrongest eclipses at the TESS period: TIC {row['ID']} (Tmag {row['Tmag']:.1f}, "
          f"{row['dstArcSec']:.1f}\" from TIC {tic})")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, period, label in ((axes[0], P, f"TESS period {P} d"), (axes[1], cand["period"],
                                                                     f"best ZTF period {cand['period']:.4f} d")):
        ref = t0 if period == P else cand["t0"]
        ph = ((t - ref) / period + 0.25) % 1 - 0.25
        ax.plot(ph, f, ".", ms=2, color="black", alpha=0.5)
        ax.set_xlabel("Phase (0 = eclipse A, 0.5 = eclipse B)")
        ax.set_title(label, fontsize=10)
    axes[0].set_ylabel("Brightness (ZTF)")
    fig.suptitle(f"ZTF light curve of TIC {row['ID']} folded")
    fig.tight_layout()
    out = OUT / f"TIC_{row['ID']}_ztf_fold.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Graph: {out.name}")


def binary_followup(tic):
    info = tic_info(tic)
    fold = run_test("1. Fold with step 2: do the dip depths alternate?", fold_binary, tic)
    run_test("2. VSX and Gaia DR3 eclipsing binaries", catalogue_check, info)
    if fold:
        P, t0, D, depth = fold
        run_test("3. Pixel test: which star dims?", pixel_test, tic, info,
                 lambda t: hunter.in_transit(t, P, t0, 0.5 * D),
                 lambda t: hunter.in_transit(t, P, t0, 3 * D) & ~hunter.in_transit(t, P, t0, 1.3 * D),
                 depth, f"TIC {tic}: where is the light lost during the eclipses?",
                 OUT / f"TIC_{tic}_pixels.png")


# ======================================================================
# TIC 356602976: the single dip
# ======================================================================
def box_scan_one(tb, fb, D):
    """Same idea as step 4's box scan, for one duration. Returns snr, depth, noise."""
    cs = np.concatenate([[0], np.cumsum(fb)])

    def mean_between(a, b):
        lo, hi = np.searchsorted(tb, a), np.searchsorted(tb, b)
        n = hi - lo
        return (cs[hi] - cs[lo]) / np.maximum(n, 1), n

    expected = D / single.BIN
    m_in, n_in = mean_between(tb - D / 2, tb + D / 2)
    m_l, n_l = mean_between(tb - 1.5 * D, tb - D / 2)
    m_r, n_r = mean_between(tb + D / 2, tb + 1.5 * D)
    ok = (n_in >= 0.8 * expected) & (n_l >= 0.5 * expected) & (n_r >= 0.5 * expected)
    if ok.sum() < 20:
        return np.zeros(tb.size), np.zeros(tb.size), np.nan
    depth = (m_l + m_r) / 2 - m_in
    noise = single.robust_noise(depth[ok])
    return np.where(ok, depth / noise, 0), depth, noise


def other_sectors(tic, t0, D, depth):
    """Download every TESS sector of the star and look for a dip like the one in sector 40."""
    import lightkurve as lk
    with quiet():
        res = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
    tab = res.table
    rank = {"SPOC": 0, "TESS-SPOC": 1, "QLP": 2}     # best data first
    best = {}
    for i in range(len(tab)):
        a, s = str(tab["author"][i]), int(tab["sequence_number"][i])
        if a in rank and (a != "SPOC" or float(tab["exptime"][i]) == 120):
            if s not in best or rank[a] < rank[str(tab["author"][best[s]])]:
                best[s] = i
    print(f"TESS observed this star in {len(best)} sectors: {sorted(best)}")
    print(f"Looking for a dip like sector 40's ({depth * 1e6:.0f} ppm, ~{D * 24:.0f} h).")
    print("  sector  source     could we see it?   strongest dips found (day: depth, SNR)")

    durations = np.array([0.35, 0.45, 0.55, 0.7])
    matches, curves = [], {}
    for s in sorted(best):
        i = best[s]
        author = str(tab["author"][i])
        try:
            with quiet():
                lc = res[i].download(quality_bitmask="default")
            if author == "QLP":
                lc = lc.select_flux("sap_flux")      # QLP's default flux is pre-smoothed and could hide the dip
            lc = lc.remove_nans()
            t = np.asarray(lc.time.value, float)
            f = np.asarray(lc.flux.value, float)
            f = f / np.nanmedian(f)
        except Exception as e:
            print(f"  {s:5d}   {author:9s}  download failed ({e.__class__.__name__})")
            continue
        flat = hunter.detrend(t, f, np.zeros(t.size, bool), window=single.WINDOW)
        tb, fb = hunter.bin_data(t, flat, size=single.BIN)
        curves[s] = (tb, fb)
        events, sens = [], np.nan
        for Dk in durations:
            snr, dep, noise = box_scan_one(tb, fb, Dk)
            if Dk == 0.55:
                sens = depth / noise                 # SNR the sector-40 dip would have here
            for j in np.argsort(snr)[::-1][:30]:
                if snr[j] > 4:
                    events.append((snr[j], tb[j], Dk, dep[j]))
        events.sort(reverse=True)
        kept = []
        for e in events:
            if all(abs(e[1] - k[1]) > (e[2] + k[2]) / 2 for k in kept):
                kept.append(e)
        for e in kept:
            if e[0] >= 5 and 0.5 < e[3] / depth < 2 and s != SECTOR:
                matches.append((s, *e))
        dips = ", ".join(f"{e[1]:.2f}: {e[3] * 1e6:.0f} ppm, SNR {e[0]:.1f}" for e in kept[:3]) or "none above SNR 4"
        print(f"  {s:5d}   {author:9s}  SNR {sens:4.1f} {'(yes)' if sens > 7 else '(weak)'}     {dips}")

    print()
    if matches:
        print("Dips in OTHER sectors with similar depth (0.5-2x) and SNR >= 5:")
        for s, snr, tc, Dk, dep in matches:
            print(f"   sector {s}: day {tc:.2f}, {dep * 1e6:.0f} ppm, {Dk * 24:.0f} h, SNR {snr:.1f}"
                  f"  -> {tc - t0:+.1f} days from the sector-40 dip")
    else:
        print("No similar dip in any other sector.")

    # one strip per sector, so you can look for yourself
    n = len(curves)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.4 * n + 1), squeeze=False)
    for ax, (s, (tb, fb)) in zip(axes[:, 0], sorted(curves.items())):
        ax.plot(tb, fb, ".", ms=1.5, color="black" if s != SECTOR else "red")
        ax.set_ylim(1 - 4 * depth, 1 + 3 * depth)
        ax.text(0.005, 0.85, f"S{s}", transform=ax.transAxes, fontsize=8)
        for m in matches:
            if m[0] == s:
                ax.axvspan(m[2] - m[3] / 2, m[2] + m[3] / 2, color="orange", alpha=0.3)
        if s == SECTOR:
            ax.axvspan(t0 - D / 2, t0 + D / 2, color="red", alpha=0.2)
    axes[-1, 0].set_xlabel("Time (days)")
    fig.suptitle(f"TIC {tic}: every sector (30-min bins). Red = sector 40 dip, orange = similar dips")
    fig.tight_layout()
    out = OUT / f"TIC_{tic}_all_sectors.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Graph: {out.name}")
    return matches


def robustness(tic, t0, D):
    """
    Is the dip still there if we clean the light curve a bit differently?
    A real dip should keep roughly the same depth and SNR; a dip made by the cleaning
    (or by the star's own wiggles) changes a lot or vanishes.
    """
    t, f, tmag, radius = survey.download(tic, SECTOR)
    print(f"  cleaning window   dip protected?   depth at day {t0}   SNR ({D * 24:.0f}-h box)")
    for window in (1.0, 1.5, 2.0, 3.0):
        for protect in (True, False):
            mask = np.abs(t - t0) < 0.75 * D if protect else np.zeros(t.size, bool)
            flat = hunter.detrend(t, f, mask, window=window)
            tb, fb = hunter.bin_data(t, flat, size=single.BIN)
            snr, dep, noise = box_scan_one(tb, fb, min(D, 0.5))
            j = np.argmin(np.abs(tb - t0))
            print(f"     {window:3.1f} d           {'yes' if protect else 'no ':3s}             "
                  f"{dep[j] * 1e6:5.0f} ppm            {snr[j]:5.1f}")


def local_depth(t, f, t0, D):
    """Depth of a dip vs. a straight line fitted just outside it. Returns depth, error."""
    dt = t - t0
    inn = np.abs(dt) < 0.4 * D
    out = (np.abs(dt) > 0.8 * D) & (np.abs(dt) < 2.2 * D)
    if inn.sum() < 10 or out.sum() < 20:
        return np.nan, np.nan
    line = np.polyval(np.polyfit(dt[out], f[out], 1), dt)
    resid = f[out] / line[out] - 1
    # noise of 30-min averages, so slow wiggles count properly
    idx = np.floor((t[out] - t[out].min()) / single.BIN).astype(int)
    cnt = np.bincount(idx)
    binned = np.bincount(idx, resid)[cnt > 0] / cnt[cnt > 0]
    n_bins = max(0.8 * D / single.BIN, 1)
    return 1 - np.mean(f[inn] / line[inn]), np.std(binned) / np.sqrt(n_bins)


def raw_flux_check(tic, t0, D):
    """SAP (raw) vs PDCSAP (cleaned) flux, background light and quality flags near the dip."""
    import lightkurve as lk
    with quiet():
        lc = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", exptime=120,
                                  sector=SECTOR)[0].download(quality_bitmask="none")
    t = lc.time.value
    q = np.asarray(lc.quality.value, int)
    sap, pdc, bkg = (np.asarray(lc[c].value, float) for c in ("sap_flux", "pdcsap_flux", "sap_bkg"))
    win = np.abs(t - t0) < 1.5

    flagged = win & (q != 0)
    print(f"Quality flags within 1.5 days of the dip: {flagged.sum()} of {win.sum()} points flagged")
    for flag in np.unique(q[flagged]):
        times = t[flagged & (q == flag)]
        print(f"   flag {flag:6d} x{times.size:4d}  days {times.min():.2f}-{times.max():.2f}: "
              f"{', '.join(lk.utils.TessQualityFlags.decode(flag))}")
    in_dip = flagged & (np.abs(t - t0) < D / 2)
    print(f"   -> flagged points INSIDE the dip itself: {in_dip.sum()}")

    ok = win & (q == 0) & np.isfinite(sap) & np.isfinite(pdc)
    d_sap, e_sap = local_depth(t[ok], sap[ok] / np.nanmedian(sap[ok]), t0, D)
    d_pdc, e_pdc = local_depth(t[ok], pdc[ok] / np.nanmedian(pdc[ok]), t0, D)
    print(f"Depth in raw SAP flux:     {d_sap * 1e6:6.0f} +- {e_sap * 1e6:.0f} ppm")
    print(f"Depth in cleaned PDCSAP:   {d_pdc * 1e6:6.0f} +- {e_pdc * 1e6:.0f} ppm")
    print("   (similar = the dip is in the raw data, not made by NASA's cleaning)")
    dt = t - t0
    b_in = np.nanmean(bkg[ok & (np.abs(dt) < 0.4 * D)])
    b_out = np.nanmean(bkg[ok & (np.abs(dt) > 0.8 * D) & (np.abs(dt) < 2.2 * D)])
    star = np.nanmedian(sap[ok])
    print(f"Background light: {b_out:.1f} e/s outside vs {b_in:.1f} e/s during the dip "
          f"(change = {(b_in - b_out) / star * 1e6:+.0f} ppm of the star's light)")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, y, name in ((axes[0], sap, "SAP (raw) flux"), (axes[1], pdc, "PDCSAP (cleaned) flux"),
                        (axes[2], bkg, "Background (e/s)")):
        good = win & (q == 0) & np.isfinite(y)
        ax.plot(t[good] - t0, y[good] / (np.nanmedian(y[good]) if ax is not axes[2] else 1), ".",
                ms=1, color="gray")
        tb, yb = hunter.bin_data(t[good] - t0, y[good] / (np.nanmedian(y[good]) if ax is not axes[2] else 1),
                                 size=single.BIN)
        ax.plot(tb, yb, "o", ms=3, color="red")
        bad = win & (q != 0)
        for x in t[bad] - t0:
            ax.axvline(x, color="orange", alpha=0.05)
        ax.axvspan(-D / 2, D / 2, color="red", alpha=0.08)
        ax.set_ylabel(name, fontsize=9)
    axes[-1].set_xlabel(f"Days from {t0}  (orange lines = flagged points)")
    fig.suptitle(f"TIC {tic}: raw vs cleaned flux and background around the dip")
    fig.tight_layout()
    out = OUT / f"TIC_{tic}_raw_flux.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Graph: {out.name}")


def neighbours_same_time(tic, t0, D, n_control=40, seed=1):
    """
    Do the other sector-40 stars we've downloaded (saved in lightkurve's cache) also dip
    at t0? Compared with random other moments, to know what's normal.
    """
    from astropy.io import fits
    cache = Path.home() / ".cache" / "lightkurve" / "mastDownload" / "TESS"
    files = sorted(cache.glob(f"*s{SECTOR:04d}*/*lc.fits"))
    rng = np.random.default_rng(seed)
    control_times = rng.uniform(2391.5, 2418.0, n_control)

    me = None
    stars = []                                  # (tic, camera, ccd, sigma at t0, sigmas at controls, curve)
    for path in files:
        try:
            with fits.open(path, memmap=False) as h:
                hdr, d = h[0].header, h[1].data
                t, f, q = np.array(d["TIME"], float), np.array(d["PDCSAP_FLUX"], float), np.array(d["QUALITY"])
            star_tic, cam, ccd = int(hdr["TICID"]), hdr["CAMERA"], hdr["CCD"]
        except Exception:
            continue
        good = np.isfinite(t) & np.isfinite(f) & (q == 0)
        t, f = t[good], f[good] / np.median(f[good])
        if star_tic == tic:
            me = (cam, ccd)
            continue
        d0, e0 = local_depth(t, f, t0, D)
        if not np.isfinite(d0):
            continue
        ctrl = [local_depth(t, f, tc, D) for tc in control_times]
        ctrl = np.array([d / e for d, e in ctrl if np.isfinite(d)])
        near = np.abs(t - t0) < 1.5
        stars.append((star_tic, cam, ccd, d0 / e0, ctrl, d0, t[near] - t0, f[near]))

    if me is None:
        me = (None, None)
    sig0 = np.array([s[3] for s in stars])
    ctrl_frac = np.mean([np.mean(s[4] > 3) for s in stars if s[4].size])
    print(f"{len(stars)} other sector-40 stars with data at day {t0} "
          f"(TIC {tic} is on camera {me[0]}, CCD {me[1]})")
    print(f"Stars dipping by > 3 sigma at day {t0}: {np.sum(sig0 > 3)} of {len(stars)} "
          f"({np.mean(sig0 > 3) * 100:.1f}%)")
    print(f"Same at {n_control} random other moments: {ctrl_frac * 100:.1f}% on average")
    same = [s for s in stars if (s[1], s[2]) == me]
    if same:
        print(f"Same camera and CCD: {len(same)} stars, {sum(s[3] > 3 for s in same)} dip by > 3 sigma; "
              f"median depth {np.median([s[5] for s in same]) * 1e6:+.0f} ppm (target: ~1250)")
    for s in sorted(stars, key=lambda s: -s[3])[:5]:
        print(f"   strongest: TIC {s[0]} (cam {s[1]}, CCD {s[2]}): {s[5] * 1e6:+.0f} ppm, {s[3]:.1f} sigma")

    # stack of same-CCD stars (or all stars) around t0
    group = same if len(same) >= 5 else stars
    grid = np.arange(-1.5, 1.5, single.BIN)
    stack = []
    for s in group:
        x, y = s[6], s[7]                       # hours-from-t0 (in days) and brightness
        if x.size < 100:
            continue
        idx = np.floor((x + 1.5) / single.BIN).astype(int)
        keep = (idx >= 0) & (idx < grid.size)
        cnt = np.bincount(idx[keep], minlength=grid.size)
        val = np.bincount(idx[keep], y[keep], minlength=grid.size)
        stack.append(np.where(cnt > 0, val / np.maximum(cnt, 1), np.nan))
    fig, ax = plt.subplots(figsize=(10, 4))
    if stack:
        med = np.nanmedian(np.array(stack), axis=0)
        ax.plot(grid * 24, med / np.nanmedian(med), "o-", ms=3, color="black",
                label=f"median of {len(stack)} other stars ({'same CCD' if group is same else 'all CCDs'})")
    ax.axvspan(-D / 2 * 24, D / 2 * 24, color="red", alpha=0.1, label="TIC 356602976's dip")
    ax.axhline(1 - 1252e-6, color="red", ls="--", lw=0.8, label="target's dip depth")
    ax.set_xlabel(f"Hours from day {t0}")
    ax.set_ylabel("Brightness")
    ax.legend(fontsize=8)
    ax.set_title("Do other stars dim at the same moment? (flat line = no)")
    fig.tight_layout()
    out = OUT / f"TIC_{tic}_neighbours.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Graph: {out.name}")


def single_followup(tic, t0, D, depth):
    info = tic_info(tic)
    print(f"TIC {tic}: Tmag {info['tmag']:.1f}, RA {info['ra']:.4f}, Dec {info['dec']:+.4f}")
    run_test("0. Does the dip survive different cleaning?", robustness, tic, t0, D)
    run_test("1. Every other TESS sector: is there a second dip?", other_sectors, tic, t0, D, depth)
    run_test("2. Raw flux, background and quality flags", raw_flux_check, tic, t0, D)
    run_test("3. Do other stars dip at the same moment?", neighbours_same_time, tic, t0, D)
    run_test("4. Pixel test: which star dims?", pixel_test, tic, info,
             lambda t: np.abs(t - t0) < 0.4 * D,
             lambda t: (np.abs(t - t0) > 0.8 * D) & (np.abs(t - t0) < 2.2 * D),
             depth, f"TIC {tic}: where is the light lost during the dip at day {t0}?",
             OUT / f"TIC_{tic}_pixels.png")


def main():
    OUT.mkdir(exist_ok=True)
    print("=" * 70 + "\nTIC 41602482: suspected eclipsing binary\n" + "=" * 70)
    binary_followup(41602482)
    print("\n" + "=" * 70 + "\nTIC 356602976: single dip at day 2414.35\n" + "=" * 70)
    single_followup(356602976, t0=2414.35, D=0.55, depth=1252e-6)


if __name__ == "__main__":
    main()
