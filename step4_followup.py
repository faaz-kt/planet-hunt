"""
Planet hunt, step 4 follow-up: done automatically for every CANDIDATE and REVIEW star.

    1. ExoFOP: is the star already a TOI or CTOI? Any confirmed planets?
    2. Catalogues: VSX / TESS EB / Kepler EB (by position) and Gaia DR3 eclipsing
       binaries (by Gaia ID).
    3. Other TESS sectors: is there a second dip of the same depth and length? If so,
       which orbital periods fit? A period only "fits" if it doesn't predict another
       transit at a time when TESS was watching and saw nothing.
    4. Binary-looking stars: ZTF light curves of the star and its neighbours within
       30", to see which star is really varying (ZTF pixels are 1", TESS pixels 21").

Called from step4_single_transits.py; follow_up() returns markdown text for the report.
"""
import contextlib
import io
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import step2_hunt as hunter
import step4_single_transits as s4

warnings.filterwarnings("ignore")

RANK = {"SPOC": 0, "TESS-SPOC": 1, "QLP": 2, "TGLC": 3, "GSFC-ELEANOR-LITE": 4}   # best data first
MAX_ZTF_STARS = 5                                  # nearest stars checked in ZTF (it's slow)
SAME_LENGTH = 0.3                                  # a second dip must be within +-30% of the first's length
MAX_DURATION_EXCESS = 1.2                          # drop periods where the dip lasts > 1.2 x the longest
                                                   # possible (central) transit for this star


def t_max_hours(period, radius, mass):
    """
    Longest possible transit (through the middle of the star) for this period:
        T_max ~ 13 h x (P / 365 d)^(1/3) x (star density / Sun's)^(-1/3),  density = M / R^3
    Off-centre transits are shorter, so a dip LONGER than this can't be that period.
    """
    density = mass / radius ** 3
    return 13.0 * (period / 365.25) ** (1 / 3) * density ** (-1 / 3)


def shortest_period(duration_h, radius, mass):
    """Shortest period that still allows this transit length (within MAX_DURATION_EXCESS)."""
    density = mass / radius ** 3
    return 365.25 * (duration_h / (MAX_DURATION_EXCESS * 13.0 * density ** (-1 / 3))) ** 3


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def star_info(tic, info):
    """Position, Gaia ID, radius and mass (Sun = 1) from the TIC. Mass ~ radius if the TIC has none."""
    if info and info.get("ra") is not None and "rad" in info:
        return info
    from astroquery.mast import Catalogs
    row = Catalogs.query_criteria(catalog="TIC", ID=tic)[0]

    def num(col):
        return None if np.ma.is_masked(row[col]) or not np.isfinite(float(row[col])) else float(row[col])

    rad = num("rad") or 1.0
    return {"ra": float(row["ra"]), "dec": float(row["dec"]), "tmag": float(row["Tmag"]),
            "gaia": "" if np.ma.is_masked(row["GAIA"]) else str(row["GAIA"]),
            "rad": rad, "mass": num("mass") or rad, "mass_from_tic": num("mass") is not None}


# ----------------------------------------------------------------------
# 1 and 2: catalogues
# ----------------------------------------------------------------------
def exofop(tic):
    import requests
    try:
        j = requests.get(f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic}&json", timeout=60).json()
    except Exception as e:
        return f"ExoFOP: lookup failed ({e.__class__.__name__})"
    tois = [f"{x.get('toi')} ({x.get('disp_tfop') or x.get('disp_tess') or '?'})" for x in (j.get("tois") or [])]
    ctois = [str(x.get("ctoi", "?")) for x in (j.get("ctois") or [])]
    planets = (j.get("basic_info") or {}).get("confirmed_planets") or ""
    text = f"ExoFOP: TOIs: {', '.join(tois) or 'none'}; CTOIs: {', '.join(ctois) or 'none'}"
    return text + (f"; confirmed planets: {planets}" if planets else "")


def gaia_eb(gaia_id):
    if not gaia_id:
        return "Gaia DR3 eclipsing binaries: no Gaia ID"
    try:
        from astroquery.gaia import Gaia
        with quiet():
            r = Gaia.launch_job("SELECT source_id, frequency FROM gaiadr3.vari_eclipsing_binary "
                                f"WHERE source_id = {gaia_id}").get_results()
    except Exception as e:
        return f"Gaia DR3 eclipsing binaries: lookup failed ({e.__class__.__name__})"
    if len(r) == 0:
        return "Gaia DR3 eclipsing binaries: not listed"
    f = r["frequency"][0]
    return "Gaia DR3 eclipsing binaries: LISTED" + (f", P = {1 / f:.4f} d" if f else "")


# ----------------------------------------------------------------------
# 3: other sectors
# ----------------------------------------------------------------------
def fetch_sector(job):
    """Runs in a helper process: one sector of one star, normalised. Returns (sector, author, t, f, error)."""
    tic, sector, author, exptime = job
    import lightkurve as lk
    try:
        with quiet(), contextlib.redirect_stderr(io.StringIO()):
            res = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author=author, sector=sector, exptime=exptime)
            lc = res[0].download(quality_bitmask="default")
        if author == "QLP":
            lc = lc.select_flux("sap_flux")      # QLP's default flux is pre-smoothed and could hide a dip
        lc = lc.remove_nans()
        t, f = np.asarray(lc.time.value, float), np.asarray(lc.flux.value, float)
        return sector, author, t, f / np.nanmedian(f), None
    except Exception as e:
        return sector, author, None, None, e.__class__.__name__


def possible_periods(t0, t1, D, depth, curves, max_n=200, min_period=1.0):
    """
    Two dips at t0 and t1 allow periods (t1 - t0) / n. Keep only periods that don't
    predict a transit where TESS has data but saw no dip (less than half the depth).
    Returns [(period, n, number of predicted transits that were checked)].
    """
    dt = abs(t1 - t0)
    allowed = []
    for n in range(1, max_n + 1):
        P = dt / n
        if P < max(min_period, 2 * D):
            break
        fits, checked = True, 0
        for tb, snr, dep in curves.values():
            for k in range(int(np.ceil((tb.min() - t0) / P)), int(np.floor((tb.max() - t0) / P)) + 1):
                tk = t0 + k * P
                if abs(tk - t0) < D or abs(tk - t1) < D:
                    continue
                j = int(np.argmin(np.abs(tb - tk)))
                if abs(tb[j] - tk) > D / 2 or snr[j] == 0:
                    continue                         # TESS wasn't watching then
                checked += 1
                if dep[j] < 0.5 * depth:
                    fits = False
                    break
            if not fits:
                break
        if fits:
            allowed.append((P, n, checked))
    return allowed


def allowed_single(t0, D, depth, curves, p_min):
    """
    Periods a SINGLE dip still allows: every period from p_min up, except those that would
    have put another transit at a time when TESS was watching and saw no dip.
    Returns (list of allowed (from, to) ranges in days; to = None means "and longer").
    """
    span = max(max(abs(tb.max() - t0), abs(tb.min() - t0)) for tb, _, _ in curves.values())

    def ruled_out(P):
        for tb, snr, dep in curves.values():
            for k in range(int(np.ceil((tb.min() - t0) / P)), int(np.floor((tb.max() - t0) / P)) + 1):
                if k == 0:
                    continue
                tk = t0 + k * P
                j = int(np.argmin(np.abs(tb - tk)))
                if abs(tb[j] - tk) <= D / 2 and snr[j] != 0 and dep[j] < 0.5 * depth:
                    return True
        return False

    ranges, start, P = [], None, p_min
    step = 1 + D / (4 * span)           # fine enough that a predicted transit can't slip past the data
    while P < span:
        if not ruled_out(P):
            start = P if start is None else start
        elif start is not None:
            ranges.append((start, P))
            start = None
        P *= step
    ranges.append((start if start is not None else max(span, p_min), None))
    return ranges


def second_dip(tic, sector, t0, D, depth, star=None):
    """Look for the same dip in every other TESS sector of this star. Returns text lines."""
    star = star or {"rad": 1.0, "mass": 1.0, "mass_from_tic": False}
    import lightkurve as lk
    with quiet():
        tab = lk.search_lightcurve(f"TIC {tic}", mission="TESS").table
    best = {}
    for i in range(len(tab)):
        author, s, exp = str(tab["author"][i]), int(tab["sequence_number"][i]), float(tab["exptime"][i])
        if author not in RANK or (author == "SPOC" and exp != 120):
            continue
        if s not in best or RANK[author] < RANK[best[s][0]]:
            best[s] = (author, 120 if author == "SPOC" else None)
    if not best:
        return ["no TESS light curves found"]
    jobs = [(tic, s, a, e) for s, (a, e) in sorted(best.items())]
    with ProcessPoolExecutor(max_workers=min(s4.PARALLEL, len(jobs))) as pool:
        results = list(pool.map(fetch_sector, jobs))

    # only dips of about the same length (+-30%) count as "the same planet"
    lengths = [D * k for k in (1 - SAME_LENGTH, 1 - SAME_LENGTH / 2, 1, 1 + SAME_LENGTH / 2, 1 + SAME_LENGTH)
               if D * k >= 1 / 24]
    curves, matches, lines = {}, [], [f"TESS has {len(best)} sector(s) of this star: {sorted(best)}"]
    for s, author, t, f, err in results:
        if t is None:
            lines.append(f"sector {s} ({author}): download failed ({err})")
            continue
        protect = np.abs(t - t0) < 0.75 * D if s == sector else np.zeros(t.size, bool)
        flat = hunter.detrend(t, f, protect, window=s4.clean_window(D))
        tb, fb = hunter.bin_data(t, flat, size=s4.BIN)
        snr_d, dep_d = s4.box_snr(tb, fb, D)
        if snr_d is None:
            lines.append(f"sector {s} ({author}): too little data")
            continue
        curves[s] = (tb, snr_d, dep_d)
        sensitivity = depth / s4.robust_noise(dep_d[snr_d != 0])
        found = []
        for Dk in lengths:
            snr, dep = s4.box_snr(tb, fb, Dk)
            if snr is None:
                continue
            for j in np.where((snr >= s4.SECOND_DIP_MIN_SNR) & (dep > 0.5 * depth) & (dep < 2 * depth)
                              & (np.abs(tb - t0) > D))[0]:
                found.append((float(snr[j]), float(tb[j]), Dk, float(dep[j])))
        found = s4.strongest_separate(found)
        matches += [(s, *e) for e in found]
        mark = " (the sector with the event)" if s == sector else ""
        lines.append(f"sector {s} ({author}){mark}: this dip would show at SNR {sensitivity:.0f}; "
                     + (f"{len(found)} similar dip(s): " + ", ".join(
                         f"day {e[1]:.2f} ({e[3] * 1e6:.0f} ppm, {e[2] * 24:.0f} h, SNR {e[0]:.1f})" for e in found[:3])
                        if found else "no similar dip"))
    # physics: how long can a transit last on this star?
    R, M, hours = star["rad"], star["mass"], D * 24
    p_min = shortest_period(hours, R, M)
    lines.append(f"physics: star R = {R:.2f}, M = {M:.2f} Suns{'' if star.get('mass_from_tic') else ' (mass ~ radius)'}; "
                 f"a {hours:.0f}-h transit needs P >= {p_min:.0f} d "
                 f"(longest central transit at that period: {t_max_hours(p_min, R, M):.1f} h, x{MAX_DURATION_EXCESS} allowed)")

    for s, snr, t1, Dk, dep in sorted(matches, reverse=True, key=lambda m: m[1])[:3]:
        fits = possible_periods(t0, t1, D, depth, curves)
        allowed = [p for p in fits if hours <= MAX_DURATION_EXCESS * t_max_hours(p[0], R, M)]
        dropped = [p for p in fits if p not in allowed]
        head = f"dip on day {t1:.2f} ({Dk * 24:.0f} h vs {hours:.0f} h)"
        if allowed:
            shown = ", ".join(f"{P:.2f} d" + (f" (checked {c} more)" if c else "") for P, n, c in allowed[:8])
            lines.append(f"if the {head} is the same planet: periods that fit = {shown}"
                         + (f" ... ({len(allowed)} in all)" if len(allowed) > 8 else ""))
        elif fits:
            lines.append(f"the {head} can't be the same planet: the periods that fit the timing "
                         f"({', '.join(f'{p[0]:.1f}' for p in fits[:6])} d) are too short for a {hours:.0f}-h transit")
        else:
            lines.append(f"the {head} can't be the same planet: every period predicts a transit that "
                         f"TESS should have seen")
        if dropped and allowed:
            lines.append(f"   (dropped as too short for a {hours:.0f}-h transit: "
                         f"{', '.join(f'{p[0]:.1f}' for p in dropped[:6])} d)")
    if not matches:
        lines.append("no second dip of the same depth and length anywhere")

    # if it's a single transit: which periods are still allowed by all the TESS coverage?
    ranges = allowed_single(t0, D, depth, curves, p_min)
    text = ", ".join(f"{a:.0f}-{b:.0f} d" if b else f"{a:.0f} d and longer" for a, b in ranges)
    lines.append(f"as a single transit, the periods still allowed (physics + all TESS coverage): {text}")
    return lines


# ----------------------------------------------------------------------
# 4: which star is varying? (ZTF)
# ----------------------------------------------------------------------
def ztf_which_star(tic, ra, dec):
    import step5_followup as f5
    import step6_vsx_check as s6
    neigh = f5.tic_neighbours(ra, dec, s4.ZTF_RADIUS_ARCSEC)
    neigh = [n for n in neigh if float(n["Tmag"]) < 17.5][:MAX_ZTF_STARS]
    lines, stats = [], []
    for n in neigh:
        nid, tmag = int(n["ID"]), float(n["Tmag"])
        label = f"TIC {nid} (Tmag {tmag:.1f}, {float(n['dstArcSec']):.1f}\" away)"
        if tmag < 12.5:
            lines.append(f"{label}: too bright for ZTF (saturates)")
            continue
        try:
            df = s6.ztf_get(nid, float(n["ra"]), float(n["dec"]))
        except Exception as e:
            lines.append(f"{label}: ZTF failed ({e.__class__.__name__})")
            continue
        if df is None or len(df) < 50:
            lines.append(f"{label}: no usable ZTF data")
            continue
        pos = df.groupby("oid")[["ra", "dec"]].median()
        sep = np.hypot((pos["ra"] - float(n["ra"])) * np.cos(np.radians(float(n["dec"]))),
                       pos["dec"] - float(n["dec"])) * 3600
        df = df[df["oid"].isin(sep[sep < 1.0].index)].reset_index(drop=True)
        if len(df) < 50:
            lines.append(f"{label}: no usable ZTF data on this star")
            continue
        df = s6.normalise(df)
        band = "r" if (df["band"] == "r").sum() >= 50 else "g"
        dm = df.loc[df["band"] == band, "dmag"].to_numpy()
        noise = s4.robust_noise(dm)
        faint = float(np.mean(dm > 4 * noise))       # how often the star is much fainter than usual
        stats.append((faint, label, df))
        lines.append(f"{label}: {dm.size} ZTF {band} points, scatter {noise * 1000:.0f} mmag, "
                     f"{faint * 100:.1f}% of points much fainter than usual")
    for faint, label, df in sorted(stats, key=lambda x: -x[0])[:2]:
        if faint < 0.01:
            continue
        P, power = s6.wide_search(df["hjd"].to_numpy(), df["flux"].to_numpy(), df["ferr"].to_numpy())
        peaks = s6.top_peaks(P, power, 3)
        lines.append(f"{label}: strongest ZTF periods " + ", ".join(f"{p:.4f} d" for p, _ in peaks))
    if stats:
        top = max(stats, key=lambda x: x[0])
        if top[0] >= 0.01 and all(s[0] < 0.4 * top[0] for s in stats if s is not top):
            lines.append(f"=> most likely the varying star: {top[1]}")
        else:
            lines.append("=> no single clearly varying star in ZTF")
    return lines


# ----------------------------------------------------------------------
def follow_up(row, info):
    """All follow-up for one star. Returns markdown text."""
    tic = int(row["tic"])
    info = star_info(tic, info)
    t0, D, depth = float(row["time"]), float(row["duration_hours"]) / 24, float(row["depth_ppm"]) / 1e6
    known = row.get("known_as") or s4.known_variable(info["ra"], info["dec"])
    out = [f"- {exofop(tic)}", f"- {gaia_eb(info.get('gaia'))}", f"- VSX / TESS EB / Kepler EB (5\"): {known}",
           "- Other TESS sectors:"]
    out += [f"    - {line}" for line in second_dip(tic, int(row.get("sector") or s4.SECTOR), t0, D, depth, info)]
    if str(row.get("binary_like")) == "True":
        out.append(f"- Looks like it could be a binary, so ZTF check of the star and neighbours within "
                   f"{s4.ZTF_RADIUS_ARCSEC}\":")
        out += [f"    - {line}" for line in ztf_which_star(tic, info["ra"], info["dec"])]
    return "\n".join(out)
