"""
Planet hunt, step 4: look for SINGLE transits, the ones big pipelines tend to miss.

Why this is where amateurs can win
    A planet with a long year (say 50+ days) may cross its star only ONCE during a
    27-day TESS sector. Period-searching pipelines (and our step 2) need at least two
    dips, so a lone dip often gets ignored. Long-period planets are also the most
    interesting ones: cooler, further out, more like our own solar system.

How it works
    1. Pick bright stars in one sector that aren't known planets, candidates or
       eclipsing binaries (checked BEFORE downloading).
    2. Download each light curve (with raw SAP flux, background and quality flags).
    3. Clean it and slide boxes of 2 h to 30 h along it; the strongest dip is the event.
       The "upside-down test" (same search for bumps) shows how big noise gets.
    4. Run every check on the event. Each check is HARD or SOFT (see SETTINGS):
           CANDIDATE = passes every check
           REVIEW    = fails only soft checks (borderline, worth a look)
           REJECT    = fails at least one hard check
       The failed checks are written to the log and onto the graph.
    5. Follow up every CANDIDATE and REVIEW automatically (step4_followup.py) and
       write a report in the "reports" folder.

Run:   python step4_single_transits.py        (safe to stop and restart)
Test:  python test_step4.py                   (regression stars + injection-recovery)
"""
import contextlib
import csv
import io
import json
import random
import textwrap
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save graphs without opening windows
import numpy as np

import step2_hunt as hunter
import step3_survey as survey

warnings.filterwarnings("ignore")

# ======================================================================
# SETTINGS: everything you might want to tune is here
# ======================================================================
# ---- the run ----------------------------------------------------------
SECTOR = 40                # which TESS sector to search
N_STARS = 200              # stars per run
PARALLEL = 6               # stars processed at the same time (set to 1 if you get errors)
MAX_TMAG = 12.0            # skip stars fainter than this (checked before downloading)
DOWNLOAD_TRIES = 3         # network trouble: try this many times...
RETRY_WAIT = 10            # ...waiting 10 s, then 20 s, ... between tries

# ---- the search -------------------------------------------------------
BIN = 30 / 60 / 24         # 30-minute bins
DURATIONS = np.array([2, 3, 4, 6, 8, 12, 16, 20, 24, 30]) / 24   # box lengths to try (days)
MIN_CLEAN_WINDOW = 1.5     # cleaning window (days) for short boxes...
CLEAN_WINDOW_FACTOR = 3.0  # ...and at least 3x the box length for long ones
MIN_SNR = 7.5              # weaker dips = "nothing found"
BUMP_MARGIN = 1.5          # the dip must be this many times stronger than the biggest bump
MAX_DIPS_CHECKED = 3       # if the strongest dip is REJECTed (often an artefact at the sector's
                           # start or end), check the next ones too, so it can't hide a real transit

# ---- HARD checks: failing any ONE of these -> REJECT ------------------
MAX_WIGGLE = 8.0           # star too variable to search (30-min wiggles vs pure noise; typical 1.1)
HARD_FLAG_BITS = 1 | 2 | 4 | 8 | 16 | 32 | 256
                           # quality flags that may not occur INSIDE the dip: attitude tweak,
                           # safe mode, coarse/Earth pointing, glint, thruster firing (momentum
                           # dump), discontinuity. ("Manual exclude" inside the dip is only soft:
                           # NASA uses it for many things, and it hit a real planet, TOI-2010 b.)
ROBUST_HARD_SNR = 4.5      # re-cleaned with other windows, SNR drops below this...
ROBUST_HARD_RATIO = 0.4    # ...or the depth changes by more than this factor
SAP_HARD_FRACTION = 0.3    # dip in raw SAP flux is less than 30% as deep as in cleaned flux
BKG_HARD_SIGMA = 4.0       # background bump during the dip is this unusual for THIS star
                           # (compared with background bumps at all other times in the sector)...
BKG_HARD_RATIO = 1.0       # ...AND bigger than the dip itself (both in electrons/s in the aperture).
                           # A plain "background rises by > half the dip" rule rejected 15% of real
                           # (injected) transits: TESS's background wanders that much all the time.
V_SHAPE_LIMIT = 9.0        # V fits this much better than a flat-bottomed transit = "V-shaped"
V_DEEP_DEPTH = 0.005       # V-shaped AND deeper than 0.5% -> eclipsing binary
                           # (only soft if the dip is next to a data gap: half a dip looks pointy)
REPEAT_SNR_FRACTION = 0.6  # other dips at least 60% as strong, similar depth and length = repeats
FLARE_SIGMA = 6.0          # a brightening this far above the noise right beside the dip, bigger than
                           # half the dip: a flare makes the normal light next to it look like a dip
                           # (TIC 141533801)
COMMON_HARD_FRACTION = 0.5 # this fraction of other stars on the same camera/CCD dip at the same time
                           # in CLEANED flux (> 3 sigma each) -> spacecraft, not the star
COMMON_RAW_HARD_FRACTION = 0.6  # ...or this fraction dip in RAW (SAP) flux AND their median glitch is at
                           # least as big as the dip (NASA's cleaning removes most glitches from most
                           # stars, so they often only show in the others' raw flux: day ~2412, sector 40).
                           # Costs 0.7% of real (injected) transits.
COMMON_MIN_STARS = 10      # need at least this many reference stars on the CCD, else the check is skipped

# ---- SOFT checks: failing only these -> REVIEW ------------------------
SOFT_SNR = 9.0             # borderline: SNR between MIN_SNR and this
ROBUST_SOFT_SNR = 6.0      # re-cleaned SNR drops below this...
ROBUST_SOFT_RATIO = 0.7    # ...or depth changes by more than this factor
SAP_SOFT_FRACTION = 0.7    # dip noticeably weaker in raw SAP flux
BKG_SOFT_SIGMA = 3.0       # background bump somewhat unusual...
BKG_SOFT_RATIO = 0.5       # ...and at least half the dip
DUMP_SOFT_DAYS = 0.5       # a thruster firing within this many days of the dip
SOFT_FLAG_BITS = 128       # "manual exclude" points inside the dip
MAX_LOPSIDED = 6.0         # way in doesn't mirror the way out
VERY_DEEP = 0.03           # deeper than 3%
MAX_PLANET_RJ = 2.0        # implied size bigger than 2 Jupiters
COMMON_SOFT_FRACTION = 0.3 # quite a few other stars on the CCD dip at the same time (cleaned flux)...
COMMON_RAW_SOFT_FRACTION = 0.4  # ...or in raw flux
# also soft: V-shaped but shallow, next to a data gap, box at the edge of the duration range

# ---- re-cleaning windows for the robustness check ---------------------
ROBUST_MIN_WINDOW = 0.75   # days; windows = max(0.75, 2.5 x box length) x (1, 2, 4)
ROBUST_WINDOW_FACTOR = 2.5 #   2-h dip: 0.75 / 1.5 / 3 d     12-h dip: 1.25 / 2.5 / 5 d
ROBUST_WINDOW_STEPS = (1, 2, 4)

# ---- reference stars for the common-mode check --------------------------
REF_STARS_PER_CCD = 30     # quiet stars per camera/CCD, taken from light curves already downloaded
                           # (built automatically, saved in cache/reference_S<sector>.pkl)

# ---- before downloading -----------------------------------------------
REMOVE_KNOWN_BINARIES = True   # TESS EB catalogue, Gaia DR3 eclipsing binaries, VSX (types E...)
PLANET_LIST_MAX_AGE = 3        # days before re-downloading the planet/candidate lists
EB_LIST_MAX_AGE = 30           # days before re-downloading the TESS EB catalogue

# ---- automatic follow-up ----------------------------------------------
FOLLOW_UP = True               # follow up every CANDIDATE and REVIEW
FOLLOW_UP_BINARY_REJECTS = False   # also check ZTF for stars REJECTed as binaries (slow)
SECOND_DIP_MIN_SNR = 5.0       # a dip in another sector counts if at least this strong
ZTF_RADIUS_ARCSEC = 30         # neighbours to check in ZTF for binary-looking stars
# ======================================================================

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "single_transits"
REPORTS = HERE / "reports"
LOG = HERE / "single_log.csv"
CACHE = HERE / "cache"
WINDOW = MIN_CLEAN_WINDOW      # old name, still used by step 5
VIZIER = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"   # Harvard mirror (Strasbourg gave errors)
VARIABLE_CATALOGS = {"B/vsx/vsx": "VSX", "J/ApJS/258/16/tess-ebs": "TESS EB catalogue",
                     "J/AJ/151/68/catalog": "Kepler EB catalogue"}
FIELDS = ["tic", "sector", "tmag", "star_radius", "n_points", "level", "hard_fails", "soft_fails",
          "time", "duration_hours", "depth_ppm", "snr", "bump_snr", "radius_earths", "period_guess_days",
          "robust_snrs", "sap_ratio", "bkg_ratio", "common_frac", "wiggle", "binary_like", "known_as", "graph",
          "verdict"]
LK_CACHE = Path.home() / ".cache" / "lightkurve" / "mastDownload" / "TESS"   # lightkurve's downloads
FLAG_NAMES = {1: "attitude tweak", 2: "safe mode", 4: "coarse point", 8: "Earth point", 16: "glint",
              32: "thruster firing", 128: "manual exclude", 256: "discontinuity"}
BINARY_SIGNS = ("repeats", "V-shaped", "too big for a planet", "very deep")


# ======================================================================
# Data
# ======================================================================
def as_array(x, fill=np.nan):
    """Plain float numpy array from a (possibly masked) column."""
    return np.ma.filled(np.ma.asarray(getattr(x, "value", x), dtype=float), fill)


def unpack(t, pdc, sap, bkg, quality, tmag, radius, tic=None, sector=None, camera=None, ccd=None):
    """
    Turn one light curve into what the search needs:
      t, f   time and cleaned (PDCSAP) flux of the good points (quality = 0), normalised
      sap    raw SAP flux of the same points, normalised
      bkg    background of the same points (electrons/s in the aperture)
      sap_e  typical SAP flux in electrons/s (so the dip can be compared with the background)
      flag_t, flag_q   times and quality flags of the flagged points (thruster firings etc.)
    """
    t, pdc, sap, bkg = as_array(t), as_array(pdc), as_array(sap), as_array(bkg)
    q = np.ma.filled(np.ma.asarray(getattr(quality, "value", quality)), 0).astype(int)
    has_t = np.isfinite(t)
    good = has_t & np.isfinite(pdc) & np.isfinite(sap) & np.isfinite(bkg) & (q == 0)
    flagged = has_t & (q != 0)
    pdc_med, sap_med = np.median(pdc[good]), np.median(sap[good])
    return {"t": t[good], "f": pdc[good] / pdc_med, "sap": sap[good] / sap_med, "bkg": bkg[good],
            "sap_e": float(sap_med), "flag_t": t[flagged], "flag_q": q[flagged],
            "tmag": None if tmag is None else float(tmag), "radius": None if radius is None else float(radius),
            "tic": None if tic is None else int(tic), "sector": None if sector is None else int(sector),
            "camera": None if camera is None else int(camera), "ccd": None if ccd is None else int(ccd)}


def download(tic, sector, author="SPOC", exptime=120):
    """Download one light curve, trying DOWNLOAD_TRIES times. None if TESS has no such data."""
    import lightkurve as lk
    for attempt in range(DOWNLOAD_TRIES):
        try:
            res = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author=author, exptime=exptime,
                                       sector=sector)
            if len(res) == 0:
                return None
            lc = res[0].download(quality_bitmask="none")
            m = lc.meta
            return unpack(lc.time.value, lc["pdcsap_flux"], lc["sap_flux"], lc["sap_bkg"], lc["quality"],
                          m.get("TESSMAG"), m.get("RADIUS"), m.get("TICID", tic), m.get("SECTOR", sector),
                          m.get("CAMERA"), m.get("CCD"))
        except Exception:
            if attempt == DOWNLOAD_TRIES - 1:
                raise
            time.sleep(RETRY_WAIT * 2 ** attempt)


def from_fits(path):
    """Same as download(), from a light-curve file already on disk (used by the tests)."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        d, hdr = h[1].data, h[0].header
        return unpack(d["TIME"], d["PDCSAP_FLUX"], d["SAP_FLUX"], d["SAP_BKG"], d["QUALITY"],
                      hdr.get("TESSMAG"), hdr.get("RADIUS"), hdr.get("TICID"), hdr.get("SECTOR"),
                      hdr.get("CAMERA"), hdr.get("CCD"))


# ======================================================================
# Reference stars (common-mode check)
# ======================================================================
_REFERENCE = {}


def build_reference(sector, verbose=True):
    """
    30-minute averaged light curves of up to REF_STARS_PER_CCD quiet stars per camera/CCD,
    from light curves already downloaded for this sector (lightkurve's cache). Saved in
    cache/reference_S<sector>.pkl and rebuilt when 50+ new light curves have arrived.
    Returns {"n_files": ..., "groups": {(camera, ccd): [(tic, tb, cleaned flux, raw SAP flux), ...]}}.
    """
    import pickle
    from astropy.io import fits
    path = CACHE / f"reference_S{sector}.pkl"
    files = sorted(LK_CACHE.glob(f"*s{sector:04d}*/*lc.fits"))
    if path.exists():
        ref = pickle.loads(path.read_bytes())
        if len(files) - ref["n_files"] < 50 and ref.get("version") == 2:
            return ref
    if verbose:
        print(f"Building reference stars for sector {sector} from {len(files)} saved light curves...")
    groups = {}
    for p in files:
        try:
            with fits.open(p, memmap=False) as h:
                key = (int(h[0].header["CAMERA"]), int(h[0].header["CCD"]))
            if len(groups.get(key, [])) >= REF_STARS_PER_CCD:
                continue
            data = from_fits(p)
            if data["t"].size < 1000:
                continue
            flat = detrend(data["t"], data["f"], np.zeros(data["t"].size, bool), window=MIN_CLEAN_WINDOW)
            if wiggliness(data["t"], flat) > MAX_WIGGLE:
                continue
            tb, fb = hunter.bin_data(data["t"], data["f"], size=BIN)
            _, sb = hunter.bin_data(data["t"], data["sap"], size=BIN)
            groups.setdefault(key, []).append((data["tic"], tb.astype(np.float32), fb.astype(np.float32),
                                               sb.astype(np.float32)))
        except Exception:
            continue
    ref = {"n_files": len(files), "groups": groups, "version": 2}
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(ref))
    if verbose:
        print(f"  {sum(len(v) for v in groups.values())} reference stars on {len(groups)} camera/CCDs")
    return ref


def reference(sector):
    """The reference set for a sector, loaded once per process."""
    if sector not in _REFERENCE:
        import pickle
        path = CACHE / f"reference_S{sector}.pkl"
        _REFERENCE[sector] = pickle.loads(path.read_bytes()) if path.exists() else {"groups": {}}
    return _REFERENCE[sector]


def binned_dip(tb, fb, tc, D):
    """Depth and error of a dip at tc in a 30-min binned light curve (straight line fitted beside it)."""
    dt = tb - tc
    inn = np.abs(dt) < max(0.4 * D, 0.75 * BIN)
    out = (np.abs(dt) > 0.8 * D) & (np.abs(dt) < 2.2 * D + BIN)
    if inn.sum() < 2 or out.sum() < 5:
        return np.nan, np.nan
    coef = np.polyfit(dt[out], fb[out], 1)
    resid = fb[out] - np.polyval(coef, dt[out])
    return float(-np.mean(fb[inn] - np.polyval(coef, dt[inn]))), float(np.std(resid) / np.sqrt(inn.sum()))


def common_mode(data, tc, D, column=2):
    """
    Do other stars on the same camera/CCD dip at the same time? column 2 = cleaned (PDCSAP)
    flux, 3 = raw (SAP) flux. (NASA's cleaning removes most spacecraft glitches from most
    stars, so a glitch often only shows in the OTHER stars' raw flux: day ~2412 in sector 40.)
    Returns (fraction dipping by > 3 sigma, number of stars, median dip of all of them) or None
    if there are too few reference stars.
    """
    if data.get("sector") is None or data.get("camera") is None:
        return None
    refs = reference(data["sector"])["groups"].get((data["camera"], data["ccd"]), [])
    z, depths = [], []
    for r in refs:
        if r[0] == data.get("tic"):
            continue
        d, e = binned_dip(r[1].astype(float), r[column].astype(float), tc, D)
        if np.isfinite(d) and e > 0:
            z.append(d / e)
            depths.append(d)
    if len(z) < COMMON_MIN_STARS:
        return None
    return float(np.mean(np.array(z) > 3)), len(z), float(np.median(depths))


# ======================================================================
# Search
# ======================================================================
def robust_noise(x):
    """Noise level that ignores outliers (1.4826 x median absolute deviation)."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def clean_window(D):
    """Cleaning window for a box of length D: long enough not to erase the dip."""
    return round(max(MIN_CLEAN_WINDOW, CLEAN_WINDOW_FACTOR * D), 4)


def detrend(t, f, mask, window):
    """
    Divide out the star's slow wobbles. Same idea as step 2's detrend (a small parabola
    fitted around each grid point, ignoring masked points, low outliers dropped once),
    with one safety rule: a grid point is skipped unless it has data on BOTH sides
    (at least 25% of the window's points on each side). Fitted to one side only, a
    parabola can shoot off wildly into a gap or a protected dip (that gave a "1167% deep"
    dip in testing); skipped points are filled in from their neighbours instead.
    """
    good = ~mask
    tg, fg = t[good], f[good]
    grid = np.arange(t.min(), t.max() + window / 6, window / 6)
    trend = np.full(grid.size, np.nan)
    lo_idx = np.searchsorted(tg, grid - window / 2)
    mid_idx = np.searchsorted(tg, grid)
    hi_idx = np.searchsorted(tg, grid + window / 2)
    for i, g in enumerate(grid):
        lo, mid, hi = lo_idx[i], mid_idx[i], hi_idx[i]
        n = hi - lo
        if n < 20 or min(mid - lo, hi - mid) < 0.25 * n:
            continue
        x, y = tg[lo:hi] - g, fg[lo:hi]
        coef = np.polyfit(x, y, 2)
        resid = y - np.polyval(coef, x)
        keep = resid > -3 * np.std(resid)
        if keep.sum() >= 20:
            coef = np.polyfit(x[keep], y[keep], 2)
        trend[i] = coef[-1]
    ok = np.isfinite(trend)
    if ok.sum() < 2:
        return f / np.median(fg)
    return f / np.interp(t, grid[ok], trend[ok])


def box_snr(tb, fb, D, sign=1):
    """
    One box length D, centred on every binned point: depth compared with the data just
    beside it (one box length each side), and SNR = depth / (how much such boxes vary
    in this star). sign=-1 looks for bumps. Returns (snr, depth), or (None, None).
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
    noise = robust_noise(depth[ok])
    return np.where(ok, depth / noise, 0), depth


def strongest_separate(events):
    """Strongest first; drop weaker events that overlap a stronger one."""
    events.sort(reverse=True)
    kept = []
    for e in events:
        if all(abs(e[1] - k[1]) > (e[2] + k[2]) / 2 for k in kept):
            kept.append(e)
    return kept


def scan(t, f, mask):
    """
    Clean the light curve and slide every box length along it. Each box length uses its
    own cleaning window (clean_window), so a 30-hour dip isn't erased by a 1.5-day window.
    Points where mask is True are protected from the cleaning.
    Returns (events, cleaned): events = [(snr, centre, length, depth), ...] strongest first;
    cleaned = {window: (flat, tb, fb)}.
    """
    groups = {}
    for D in DURATIONS:
        groups.setdefault(clean_window(D), []).append(float(D))
    events, cleaned = [], {}
    for w, lengths in groups.items():
        flat = detrend(t, f, mask, window=w)
        tb, fb = hunter.bin_data(t, flat, size=BIN)
        cleaned[w] = (flat, tb, fb)
        for D in lengths:
            snr, depth = box_snr(tb, fb, D)
            if snr is None:
                continue
            for i in np.argsort(snr)[::-1][:20]:
                if snr[i] > 0:
                    events.append((float(snr[i]), float(tb[i]), D, float(depth[i])))
    return strongest_separate(events), cleaned


def refine_event(t, f, event):
    """
    The search only tries 10 box lengths, on a 30-minute grid, so the best box can be
    too long or a bit off-centre (TOI-2010 b: 12 h box on an 8.7 h transit, which then
    looked V-shaped). Try lengths from 0.6x to 1.5x and centres within half a box,
    and keep the strongest. Returns the refined (snr, centre, length, depth).
    """
    snr0, tc, D, depth0 = event
    best = event
    mask = np.abs(t - tc) < 0.75 * D * 1.5
    flat = detrend(t, f, mask, window=clean_window(D * 1.5))
    tb, fb = hunter.bin_data(t, flat, size=BIN)
    for k in (0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5):
        Dk = D * k
        if not (1 / 24 <= Dk <= 1.5):
            continue
        snr, depth = box_snr(tb, fb, Dk)
        if snr is None:
            continue
        near = np.where(np.abs(tb - tc) <= D / 2)[0]
        if near.size:
            j = near[np.argmax(snr[near])]
            if snr[j] > best[0]:
                best = (float(snr[j]), float(tb[j]), float(Dk), float(depth[j]))
    return best


def biggest_bump(cleaned, tc, D):
    """The upside-down test: the strongest BUMP anywhere except at the dip."""
    best = 0.0
    for w, (flat, tb, fb) in cleaned.items():
        away = np.abs(tb - tc) > 2 * D
        for Dk in DURATIONS:
            if clean_window(Dk) == w:
                snr, _ = box_snr(tb[away], fb[away], Dk, -1)
                if snr is not None:
                    best = max(best, float(snr.max()))
    return best


def wiggliness(t, flat):
    """
    How much the cleaned light curve wiggles on 30-minute-to-day timescales, compared
    with pure random noise. ~1 for a quiet star; a pulsating star scores much higher.
    """
    tb, fb = hunter.bin_data(t, flat, size=BIN)
    point_noise = robust_noise(np.diff(flat)) / np.sqrt(2)
    return robust_noise(fb) / (point_noise / np.sqrt(BIN / np.median(np.diff(t))))


# ======================================================================
# Checks
# ======================================================================
def shape_test(tb, fb, tc, D):
    """How much better a V fits than a flat-bottomed transit, in units of noise^2."""
    near = np.abs(tb - tc) < 2.5 * D
    outside = near & (np.abs(tb - tc) > 0.75 * D)
    if outside.sum() < 5:
        return 0.0
    y, x = np.median(fb[outside]) - fb[near], tb[near] - tc
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

    return (best_chi2(lambda u: np.clip((1 - u) / 0.15, 0, 1)) - best_chi2(lambda u: np.clip(1 - u, 0, 1))) / sigma2


def lopsidedness(tb, fb, tc, D):
    """Mismatch between the way in and the mirrored way out (~1 for a symmetric dip)."""
    inner = np.abs(tb - tc) < 0.3 * D
    depth = np.median(fb[np.abs(tb - tc) > D]) - np.median(fb[inner]) if inner.any() else 0.0
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


def robust_windows(D):
    base = max(ROBUST_MIN_WINDOW, ROBUST_WINDOW_FACTOR * D)
    return [round(base * k, 3) for k in ROBUST_WINDOW_STEPS]


def dip_survives(t, f, tc, D):
    """Re-measure the dip after cleaning with each robustness window. Returns SNRs, depths."""
    mask = np.abs(t - tc) < 0.75 * D
    snrs, depths = [], []
    for window in robust_windows(D):
        flat = detrend(t, f, mask, window=window)
        tb, fb = hunter.bin_data(t, flat, size=BIN)
        snr, depth = box_snr(tb, fb, D)
        j = np.argmin(np.abs(tb - tc))
        snrs.append(float(snr[j]) if snr is not None else 0.0)
        depths.append(float(depth[j]) if snr is not None else 0.0)
    return snrs, depths


def local_change(t, y, tc, D):
    """
    How much y changes during the dip, compared with a straight line fitted just outside
    it (0.8 to 2.2 box lengths away). Returns (change, error) in the units of y; the error
    uses the scatter of 30-minute averages, so slow wiggles are counted fairly.
    """
    dt = t - tc
    inn = np.abs(dt) < 0.4 * D
    out = (np.abs(dt) > 0.8 * D) & (np.abs(dt) < 2.2 * D)
    if inn.sum() < 5 or out.sum() < 20:
        return np.nan, np.nan
    coef = np.polyfit(dt[out], y[out], 1)
    resid = y[out] - np.polyval(coef, dt[out])
    idx = np.floor((t[out] - t[out].min()) / BIN).astype(int)
    cnt = np.bincount(idx)
    binned = np.bincount(idx, resid)[cnt > 0] / cnt[cnt > 0]
    return (float(np.mean(y[inn] - np.polyval(coef, dt[inn]))),
            float(np.std(binned) / np.sqrt(max(0.8 * D / BIN, 1))))


def flag_names(q):
    return ", ".join(sorted({name for bit, name in FLAG_NAMES.items() if np.any(q & bit)}))


def run_checks(data, cleaned, best, others):
    """
    Every check on the event. Returns (hard, soft, info): lists of failed checks written as
    "name (details)", and extra numbers for the log.
    """
    t, f = data["t"], data["f"]
    snr, tc, D, depth = best
    flat, tb, fb = cleaned[clean_window(D)]
    hard, soft, info = [], [], {}

    # -- strength and box length
    if snr < SOFT_SNR:
        soft.append(f"borderline SNR ({snr:.1f} < {SOFT_SNR})")
    if D >= DURATIONS.max() * 0.999:      # stuck at the longest box: the real dip may be longer still
        soft.append(f"edge of duration range ({D * 24:.0f} h box)")

    # -- does it survive other cleaning windows?
    snrs, depths = dip_survives(t, f, tc, D)
    info["robust_snrs"] = "/".join(f"{s:.1f}" for s in snrs)
    ratio = min(depths) / max(depths) if max(depths) > 0 else 0.0
    detail = (f"SNR {info['robust_snrs']} with {'/'.join(f'{w:g}' for w in robust_windows(D))}-day "
              f"cleaning, depth ratio {ratio:.2f}")
    if min(snrs) < ROBUST_HARD_SNR or ratio < ROBUST_HARD_RATIO:
        hard.append(f"changes with cleaning ({detail})")
    elif min(snrs) < ROBUST_SOFT_SNR or ratio < ROBUST_SOFT_RATIO:
        soft.append(f"changes somewhat with cleaning ({detail})")

    # -- raw SAP flux: is the dip there before NASA's cleaning?
    pdc_change, pdc_err = local_change(t, f, tc, D)
    sap_change, _ = local_change(t, data["sap"], tc, D)
    pdc_depth, sap_depth = -pdc_change, -sap_change
    if np.isfinite(pdc_depth) and pdc_depth > 2 * pdc_err:
        r = sap_depth / pdc_depth
        info["sap_ratio"] = round(r, 2)
        detail = f"raw SAP {sap_depth * 1e6:.0f} ppm vs cleaned {pdc_depth * 1e6:.0f} ppm"
        if r < SAP_HARD_FRACTION:
            hard.append(f"not in raw SAP flux ({detail})")
        elif r < SAP_SOFT_FRACTION:
            soft.append(f"weaker in raw SAP flux ({detail})")
    else:
        soft.append(f"unclear in a local measurement ({pdc_depth * 1e6:.0f} +- {pdc_err * 1e6:.0f} ppm)")

    # -- background: only a background RISE can fake a dip (too much gets subtracted).
    # Measured like the flux search: the background bump during the dip (box vs the data just
    # beside it), compared with background bumps of the same length at all other times.
    # Both the bump and the dip are in electrons/s in the star's aperture.
    tbb, bbb = hunter.bin_data(t, data["bkg"], size=BIN)
    b_snr, b_change = box_snr(tbb, bbb, D, sign=-1)
    dip_flux = depth * data["sap_e"]
    if b_snr is not None and dip_flux > 0:
        j = int(np.argmin(np.abs(tbb - tc)))
        z, r = float(b_snr[j]), float(b_change[j]) / dip_flux
        info["bkg_ratio"] = round(r, 2)
        detail = f"background {b_change[j]:+.1f} e/s vs dip {dip_flux:.1f} e/s, {z:.1f} sigma unusual for this star"
        if z > BKG_HARD_SIGMA and r > BKG_HARD_RATIO:
            hard.append(f"background rises during the dip ({detail})")
        elif z > BKG_SOFT_SIGMA and r > BKG_SOFT_RATIO:
            soft.append(f"background rises a little ({detail})")

    # -- quality flags: thruster firings etc. inside the dip, or close to it
    ft, fq = data["flag_t"], data["flag_q"]
    in_dip = np.abs(ft - tc) < D / 2
    inside = in_dip & ((fq & HARD_FLAG_BITS) != 0)
    inside_soft = in_dip & ((fq & SOFT_FLAG_BITS) != 0) & ~inside
    if inside.any():
        hard.append(f"flagged points inside the dip ({inside.sum()}: {flag_names(fq[inside])})")
    elif inside_soft.any():
        soft.append(f"manually excluded points inside the dip ({inside_soft.sum()})")
    if not inside.any():
        dumps = ft[(fq & 32) != 0]
        if dumps.size:
            gap = float(np.min(np.abs(dumps - tc)) - D / 2)
            if gap < DUMP_SOFT_DAYS:
                soft.append(f"thruster firing nearby ({max(gap, 0):.2f} d from the dip)")

    # -- do other stars on the same camera/CCD dip at the same moment? (then it's the spacecraft)
    cm_clean, cm_raw = common_mode(data, tc, D, 2), common_mode(data, tc, D, 3)
    if cm_clean and cm_raw:
        (fc, n, _), (fr, _, glitch) = cm_clean, cm_raw
        info["common_frac"] = f"{fc:.2f}/{fr:.2f}"
        detail = (f"camera {data['camera']} CCD {data['ccd']}: {round(fc * n)} of {n} stars dip in cleaned flux, "
                  f"{round(fr * n)} in raw flux, median raw glitch {glitch * 1e6:+.0f} ppm vs dip {depth * 1e6:.0f}")
        if fc >= COMMON_HARD_FRACTION or (fr >= COMMON_RAW_HARD_FRACTION and glitch >= depth):
            hard.append(f"same-time dips in other stars: spacecraft ({detail})")
        elif fc >= COMMON_SOFT_FRACTION or fr >= COMMON_RAW_SOFT_FRACTION:
            soft.append(f"same-time dips in some other stars ({detail})")

    # -- data on both sides of the dip
    near_gap = False
    for a, b in ((tc - 1.5 * D, tc - 0.5 * D), (tc + 0.5 * D, tc + 1.5 * D)):
        if np.sum((tb > a) & (tb < b)) < 0.5 * D / BIN:
            soft.append("next to a data gap (little data on one side)")
            near_gap = True
            break

    # -- a flare right beside the dip (stars brighten suddenly; transits never do)
    near = np.abs(tb - tc) < 1.5 * D
    ref = (np.abs(tb - tc) > 1.5 * D) & (np.abs(tb - tc) < 3 * D)
    if near.any() and ref.sum() > 3:
        peak = float(np.max(fb[near]) - np.median(fb[ref]))
        point_noise = robust_noise(np.diff(fb)) / np.sqrt(2)
        if peak > FLARE_SIGMA * point_noise and peak > 0.5 * depth:
            hard.append(f"flare next to the dip (brightening of {peak * 1e6:.0f} ppm, "
                        f"{peak / point_noise:.0f}x the noise)")

    # -- shape (not trusted next to a gap: half a dip looks pointy)
    v = shape_test(tb, fb, tc, D)
    if v > V_SHAPE_LIMIT:
        if depth > V_DEEP_DEPTH and not near_gap:
            hard.append(f"V-shaped and deep ({depth * 100:.2f}%): eclipsing binary?")
        else:
            soft.append(f"V-shaped (V fits better by {v:.0f})")
    lop = lopsidedness(tb, fb, tc, D)
    if lop > MAX_LOPSIDED:
        soft.append(f"lopsided ({lop:.1f})")

    # -- size
    if depth > VERY_DEEP:
        soft.append(f"very deep ({depth * 100:.1f}%)")
    if data["radius"]:
        r_earth = np.sqrt(max(depth, 0)) * data["radius"] * hunter.SUN_IN_EARTHS
        if r_earth > MAX_PLANET_RJ * hunter.EARTHS_IN_JUPITER:
            soft.append(f"too big for a planet ({r_earth / hunter.EARTHS_IN_JUPITER:.1f} Jupiters)")

    # -- does it repeat? (then it's periodic: a binary or a short-period planet, use step 2)
    similar = [o for o in others if o[0] >= REPEAT_SNR_FRACTION * snr and 0.6 < o[3] / depth < 1.6
               and 0.67 < o[2] / D < 1.5]
    if similar:
        hard.append(f"repeats ({len(similar) + 1} similar dips: periodic, use step 2)")
    return hard, soft, info


def period_guess(duration_days, star_radius):
    """Rough period from the transit length (central transit, density ~ 1/R^2)."""
    density = 1 / star_radius ** 2 if star_radius else 1.0
    return 365.25 * (duration_days * 24 / 13) ** 3 * density


def check_name(failure):
    """'changes with cleaning (SNR ...)' -> 'changes with cleaning'."""
    return failure.split(" (")[0].split(":")[0].strip()


def search(data):
    """
    Full search on one light curve. Returns (result, plot_data); result has 'level'
    (CANDIDATE / REVIEW / REJECT / NOTHING) and the failed checks.
    """
    t, f = data["t"], data["f"]
    no_mask = np.zeros(t.size, bool)
    flat0 = detrend(t, f, no_mask, window=MIN_CLEAN_WINDOW)
    wiggle = round(float(wiggliness(t, flat0)), 2)
    if wiggle > MAX_WIGGLE:
        fail = f"too variable to search (wiggles {wiggle}x random noise)"
        return {"level": "REJECT", "hard_fails": fail, "soft_fails": "", "wiggle": wiggle,
                "verdict": f"REJECT: {fail}"}, None
    events, cleaned = scan(t, f, no_mask)
    if not events:
        return {"level": "NOTHING", "wiggle": wiggle, "verdict": "nothing found"}, None

    # Protect a dip from the cleaning and look again, so it isn't partly erased. Try the
    # top 3 (the first cleaning can weaken a long dip) and keep the strongest result.
    tries = []
    for cand in events[:3]:
        ev, cl = scan(t, f, np.abs(t - cand[1]) < 0.75 * cand[2])
        if ev:
            tries.append((ev[0][0], ev, cl))
    if not tries:
        return {"level": "NOTHING", "wiggle": wiggle, "verdict": "nothing found"}, None
    _, events, cleaned = max(tries, key=lambda x: x[0])

    def describe(best, bump, cl):
        snr, tc, D, depth = best
        r = {"time": round(tc, 3), "duration_hours": round(D * 24, 1), "depth_ppm": round(depth * 1e6),
             "snr": round(snr, 1), "bump_snr": round(bump, 1), "wiggle": wiggle,
             "radius_earths": None, "period_guess_days": None, "_event": best}
        if data["radius"]:
            r["radius_earths"] = round(np.sqrt(max(depth, 0)) * data["radius"] * hunter.SUN_IN_EARTHS, 1)
            r["period_guess_days"] = round(period_guess(D, data["radius"]))
        return r, cl[clean_window(D)]

    # Check the strongest dips in turn; stop at the first one that isn't REJECTed.
    judged, rejected, first_weak = [], [], None
    for ev in events[:MAX_DIPS_CHECKED]:
        best = refine_event(t, f, ev)
        snr, tc, D, depth = best
        flat = detrend(t, f, np.abs(t - tc) < 0.75 * D, window=clean_window(D))
        cl = dict(cleaned)
        cl[clean_window(D)] = (flat, *hunter.bin_data(t, flat, size=BIN))
        bump = biggest_bump(cleaned, tc, D)
        if snr < MIN_SNR or snr < BUMP_MARGIN * bump:
            if first_weak is None:
                why = "nothing found" if snr < MIN_SNR else "nothing found (dip no bigger than the noise bumps)"
                first_weak = (best, bump, cl, why)
            continue
        others = [e for e in events if e is not ev and e not in rejected]
        hard, soft, info = run_checks(data, cl, best, others)
        level = "REJECT" if hard else "REVIEW" if soft else "CANDIDATE"
        judged.append((level, best, bump, cl, hard, soft, info))
        if level != "REJECT":
            break
        rejected.append(ev)

    if not judged:
        best, bump, cl, why = first_weak if first_weak else (refine_event(t, f, events[0]), 0.0, cleaned, "nothing found")
        if clean_window(best[2]) not in cl:
            return {"level": "NOTHING", "wiggle": wiggle, "verdict": why}, None
        result, plot_data = describe(best, bump, cl)
        result.update(level="NOTHING", verdict=why)
        return result, plot_data
    level, best, bump, cl, hard, soft, info = next((j for j in judged if j[0] != "REJECT"), judged[0])
    result, plot_data = describe(best, bump, cl)
    result.update(info)
    result.update(level=level, hard_fails="; ".join(hard), soft_fails="; ".join(soft),
                  binary_like=any(check_name(x).startswith(BINARY_SIGNS) for x in hard + soft))
    result["verdict"] = level + (": " + "; ".join(check_name(x) for x in hard + soft) if hard + soft else "")
    return result, plot_data


# ======================================================================
# Plot
# ======================================================================
def save_plot(tic, sector, data, res, plot_data, out):
    import matplotlib.pyplot as plt
    snr, tc, D, depth = res["_event"]
    flat, tb, fb = plot_data
    t = data["t"]
    colour = {"CANDIDATE": "green", "REVIEW": "darkorange", "REJECT": "red"}.get(res["level"], "black")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 3, 1.6]})
    axes[0].plot(t, flat, ",", color="black", alpha=0.4)
    axes[0].plot(tb, fb, ".", ms=2, color="gray")
    axes[0].axvspan(tc - D, tc + D, color=colour, alpha=0.15)
    for ft in data["flag_t"][(data["flag_q"] & 32) != 0]:
        axes[0].axvline(ft, color="purple", lw=0.5, alpha=0.5)
    axes[0].set_title(f"TIC {tic}, sector {sector}: whole sector (shaded = the event, purple = thruster firings)",
                      fontsize=10)
    axes[0].set_xlabel("Time (days)")

    win, winb = np.abs(t - tc) < 4 * D, np.abs(tb - tc) < 4 * D
    axes[1].plot((t[win] - tc) * 24, flat[win], ".", ms=2, color="lightgray")
    axes[1].plot((tb[winb] - tc) * 24, fb[winb], "o", ms=4, color=colour)
    sap_win = data["sap"][win]
    axes[1].plot((t[win] - tc) * 24, sap_win / np.median(sap_win) - 4 * depth, ",", color="steelblue")
    axes[1].axhline(1 - depth, color=colour, lw=0.8, ls="--")
    size = f", ~{res['radius_earths']} Earth radii" if res.get("radius_earths") else ""
    axes[1].set_title(f"Zoom: {res['duration_hours']} h, {res['depth_ppm']} ppm{size}, SNR {res['snr']} "
                      f"(biggest bump {res['bump_snr']}); blue = raw SAP flux, shifted down", fontsize=9)
    axes[1].set_xlabel("Hours from the middle of the event")

    axes[2].axis("off")
    lines = [f"{res['level']}"]
    for kind in ("hard_fails", "soft_fails"):
        for item in filter(None, (res.get(kind) or "").split("; ")):
            lines += textwrap.wrap(f"{'HARD' if kind == 'hard_fails' else 'soft'}: {item}", 140)
    if len(lines) == 1:
        lines.append("passes every check")
    axes[2].text(0, 1, lines[0], color=colour, fontsize=12, weight="bold", va="top", transform=axes[2].transAxes)
    axes[2].text(0, 0.72, "\n".join(lines[1:8]), fontsize=8, va="top", transform=axes[2].transAxes,
                 family="monospace")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ======================================================================
# One star (runs in a helper process)
# ======================================================================
def analyse(tic, data, sector=SECTOR):
    base = {"tic": tic, "sector": sector}
    if data is None:
        return dict(base, level="SKIPPED", verdict="no 2-minute data")
    base.update(tmag=data["tmag"], star_radius=data["radius"], n_points=data["t"].size)
    if data["tmag"] is not None and data["tmag"] > MAX_TMAG:
        return dict(base, level="SKIPPED", verdict=f"skipped: too faint (Tmag {data['tmag']:.1f})")
    if data["t"].size < 1000:
        return dict(base, level="SKIPPED", verdict="skipped: too little data")
    res, plot_data = search(data)
    row = dict(base, **{k: v for k, v in res.items() if not k.startswith("_")})
    if plot_data is not None and res["level"] in ("CANDIDATE", "REVIEW", "REJECT"):
        out = OUT_DIR / f"TIC_{tic}_S{sector}_single.png"
        save_plot(tic, sector, data, res, plot_data, out)
        row["graph"] = out.name
    return row


def process_star(tic):
    """Download (with retries) and search one star. Never raises: problems become an ERROR row."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            data = download(tic, SECTOR)
    except Exception as e:
        return {"tic": tic, "sector": SECTOR, "level": "ERROR",
                "verdict": f"error: download failed {DOWNLOAD_TRIES} times ({e.__class__.__name__}: {e})"[:200]}
    try:
        return analyse(tic, data)
    except Exception as e:
        where = traceback.extract_tb(e.__traceback__)[-1]
        return {"tic": tic, "sector": SECTOR, "level": "ERROR",
                "verdict": f"error: {e.__class__.__name__} in {where.name} line {where.lineno}"}


# ======================================================================
# Choosing stars
# ======================================================================
def cached(name, make, max_age_days):
    """A saved list of TIC numbers if recent enough; otherwise make() it and save it."""
    path = CACHE / f"{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        ids = set(json.loads(path.read_text()))
        print(f"  using saved list {path.name} ({len(ids):,} stars)")
        return ids
    ids = make()
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(sorted(ids)))
    return ids


def tess_eb_tics():
    """All TIC numbers in the TESS eclipsing-binary catalogue (Prsa et al. 2022)."""
    import requests
    text = requests.get(VIZIER, timeout=180, params={"-source": "J/ApJS/258/16/tess-ebs", "-out": "TIC",
                                                      "-out.max": "unlimited"}).text
    ids = {int(x.strip()) for x in text.splitlines() if x.strip().isdigit()}
    print(f"  TESS EB catalogue: {len(ids):,} stars")
    return ids


def vizier_rows(source, ra, dec, radius_arcsec, timeout=60):
    """Rows of one VizieR catalogue near a position (Harvard mirror)."""
    import requests
    text = requests.get(VIZIER, timeout=timeout, params={
        "-source": source, "-c": f"{ra} {dec:+}", "-c.rs": radius_arcsec, "-out.max": 20}).text
    rows, table, header, units, in_data = [], None, None, None, False
    for line in text.splitlines():
        if line.startswith("#Name:"):
            table, header, units, in_data = line.split(":", 1)[1].strip(), None, None, False
        elif line.startswith("#") or not line.strip():
            in_data = False
        elif header is None:
            header = [h.strip() for h in line.split("\t")]
        elif units is None:
            units = line
        elif line.startswith("-"):
            in_data = True
        elif in_data:
            rows.append(dict(zip(header, (v.strip() for v in line.split("\t"))), _table=table))
    return rows


def pick_bright_stars(fresh, n):
    """
    n random stars bright enough to search, checked BEFORE downloading (one catalogue
    query per 150 stars). Returns {tic: {"ra", "dec", "gaia", "tmag"}}.
    """
    from astroquery.mast import Catalogs
    fresh = list(fresh)
    random.shuffle(fresh)
    picks = {}
    for start in range(0, len(fresh), 150):
        if len(picks) >= n:
            break
        for row in Catalogs.query_criteria(catalog="TIC", ID=fresh[start:start + 150]):
            if len(picks) < n and not np.ma.is_masked(row["Tmag"]) and row["Tmag"] <= MAX_TMAG:
                gaia = "" if np.ma.is_masked(row["GAIA"]) else str(row["GAIA"])
                picks[int(row["ID"])] = {"ra": float(row["ra"]), "dec": float(row["dec"]),
                                         "gaia": gaia, "tmag": float(row["Tmag"])}
    return picks


def known_binaries(picks):
    """
    Which picked stars are already known eclipsing binaries? Gaia DR3's list (by Gaia ID)
    and VSX (by position, 5", types starting with E). Answers are saved in the cache
    folder, so each star is only asked about once. Returns {tic: "where it's listed"}.
    """
    path = CACHE / "binary_check.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    todo = [tic for tic in picks if str(tic) not in done]
    found, ok = {}, set(todo)
    try:
        from astroquery.gaia import Gaia
        ids = {picks[t]["gaia"]: t for t in todo if picks[t]["gaia"]}
        id_list = list(ids)
        for i in range(0, len(id_list), 300):
            q = ("SELECT source_id FROM gaiadr3.vari_eclipsing_binary WHERE source_id IN ("
                 + ",".join(id_list[i:i + 300]) + ")")
            with contextlib.redirect_stdout(io.StringIO()):
                for r in Gaia.launch_job(q).get_results():
                    found[ids[str(r["source_id"])]] = "Gaia DR3 eclipsing binary"
    except Exception as e:
        print(f"  (Gaia check failed: {e.__class__.__name__}; will try again next run)")
        ok = set()

    def vsx(tic):
        try:
            rows = vizier_rows("B/vsx/vsx", picks[tic]["ra"], picks[tic]["dec"], 5, timeout=60)
            return tic, [r.get("Type", "") for r in rows], True
        except Exception:
            return tic, [], False

    with ThreadPoolExecutor(8) as pool:
        for tic, types, success in pool.map(vsx, todo):
            if not success:
                ok.discard(tic)
            eb = [x for x in types if x.upper().startswith("E")]
            if eb:
                found[tic] = (found.get(tic, "") + "; " if tic in found else "") + f"VSX {eb[0]}"
    for tic in ok:
        done[str(tic)] = found.get(tic, "")
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(done))
    return {tic: done.get(str(tic)) or found.get(tic) for tic in picks if done.get(str(tic)) or found.get(tic)}


def known_variable(ra, dec):
    """Known variable star / binary at this position (VSX, TESS EB, Kepler EB, 5")?"""
    try:
        rows = vizier_rows(" ".join(VARIABLE_CATALOGS), ra, dec, 5, timeout=30)
    except Exception as e:
        return f"catalogue check failed ({e.__class__.__name__})"
    found = []
    for r in rows:
        name = next((label for src, label in VARIABLE_CATALOGS.items() if r["_table"] == src), None)
        if name:
            period = r.get("Period") or r.get("Per")
            found.append(name + (f" type {r['Type']}" if r.get("Type") else "")
                         + (f", P={float(period):.4f} d" if period else ""))
    return "; ".join(found) if found else "not in catalogues"


# ======================================================================
# Log and report
# ======================================================================
def read_log():
    rows = []
    for path in sorted(LOG.parent.glob(f"{LOG.stem}*.csv")):
        try:
            with path.open(newline="") as fh:
                rows += list(csv.DictReader(fh))
        except OSError:
            pass
    return rows


def done_tics():
    """Stars already searched (failed downloads don't count: they're tried again)."""
    return {int(r["tic"]) for r in read_log() if r.get("tic") and r.get("level") != "ERROR"
            and not (r.get("verdict") or "").startswith("error")}


def upgrade_log():
    """Older logs miss the newer columns; rewrite the file with them."""
    if not LOG.exists():
        return
    with LOG.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if set(FIELDS) <= set(reader.fieldnames or []):
            return
        rows = list(reader)
    with LOG.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_row(row):
    """Add one row to the log; if the file is locked (Excel, OneDrive), retry, then use a backup log."""
    for attempt in range(5):
        try:
            new = not LOG.exists()
            with LOG.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)
            return
        except PermissionError:
            time.sleep(2)
    backup = LOG.with_name(f"{LOG.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with backup.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow(row)


def write_report(rows, followups, minutes, removed):
    """A markdown report of this run (open it in VS Code with the preview to see the graphs)."""
    from collections import Counter
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"report_S{SECTOR}_{datetime.now():%Y%m%d_%H%M}.md"
    levels = Counter(r.get("level", "?") for r in rows)
    hard = Counter(check_name(x) for r in rows if r.get("level") == "REJECT"
                   for x in (r.get("hard_fails") or "").split("; ") if x)
    soft = Counter(check_name(x) for r in rows if r.get("level") == "REVIEW"
                   for x in (r.get("soft_fails") or "").split("; ") if x)
    lines = [f"# Single-transit search: sector {SECTOR}, {datetime.now():%d %b %Y %H:%M}", "",
             f"{len(rows)} stars in {minutes:.1f} minutes. Removed before downloading: "
             + ", ".join(f"{v} {k}" for k, v in removed.items()) + ".", "",
             "| Level | Stars |", "|---|---|"]
    for lvl in ("CANDIDATE", "REVIEW", "REJECT", "NOTHING", "SKIPPED", "ERROR"):
        lines.append(f"| {lvl} | {levels.get(lvl, 0)} |")
    lines += ["", "## Why stars were rejected (hard checks)", "", "| Check | Stars |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in hard.most_common()] or ["| (none) | |"]
    lines += ["", "## Why stars need review (soft checks)", "", "| Check | Stars |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in soft.most_common()] or ["| (none) | |"]
    for lvl in ("CANDIDATE", "REVIEW"):
        chosen = [r for r in rows if r.get("level") == lvl]
        lines += ["", f"## {lvl} ({len(chosen)})", ""]
        for r in chosen:
            lines += [f"### TIC {r['tic']}", "",
                      f"- day {r['time']}, {r['duration_hours']} h, {r['depth_ppm']} ppm, SNR {r['snr']}, "
                      f"~{r.get('radius_earths')} Earth radii, period maybe {r.get('period_guess_days')}+ d",
                      f"- soft fails: {r.get('soft_fails') or 'none'}",
                      f"- catalogues: {r.get('known_as', '?')}",
                      f"- ExoFOP: https://exofop.ipac.caltech.edu/tess/target.php?id={r['tic']}", ""]
            if r["tic"] in followups:
                lines += ["**Follow-up**", "", followups[r["tic"]], ""]
            if r.get("graph"):
                lines += [f"![TIC {r['tic']}](../single_transits/{r['graph']})", ""]
    errors = [r for r in rows if r.get("level") == "ERROR"]
    if errors:
        lines += ["", "## Errors (tried again next run)", ""] + [f"- TIC {r['tic']}: {r['verdict']}" for r in errors]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ======================================================================
# The run
# ======================================================================
def main():
    from collections import Counter
    OUT_DIR.mkdir(exist_ok=True)
    upgrade_log()
    t_start = time.time()
    stars = cached(f"sector{SECTOR}_targets", lambda: survey.sector_targets(SECTOR), 365)
    print("Removing stars that are known planets, candidates or eclipsing binaries:")
    known = cached("known_stars", survey.known_stars, PLANET_LIST_MAX_AGE)
    ebs = cached("tess_eb_tics", tess_eb_tics, EB_LIST_MAX_AGE) if REMOVE_KNOWN_BINARIES else set()
    done = done_tics()
    fresh = stars - known - ebs - done
    removed = {"known planets/candidates": len(stars & known), "TESS EB catalogue stars": len((stars - known) & ebs)}
    print(f"{len(fresh):,} stars to choose from ({len(stars & done):,} already searched).")
    print(f"Choosing {N_STARS} bright enough ones (Tmag <= {MAX_TMAG})...")
    info = pick_bright_stars(fresh, int(N_STARS * 1.15) + 5)
    if REMOVE_KNOWN_BINARIES:
        print("Checking them against Gaia DR3 eclipsing binaries and VSX...")
        listed = known_binaries(info)
        removed["Gaia/VSX binaries"] = len(listed)
        for tic, where in listed.items():
            print(f"  TIC {tic}: {where} -> removed")
    else:
        listed = {}
    picks = [tic for tic in info if tic not in listed][:N_STARS]
    build_reference(SECTOR)                     # reference stars for the common-mode check
    print(f"\nSearching {len(picks)} stars, {PARALLEL} at a time...")

    rows, counts, last_note = [], Counter(), time.time()
    t_search = time.time()
    with ProcessPoolExecutor(max_workers=PARALLEL) as pool:
        jobs = {pool.submit(process_star, tic): tic for tic in picks}
        for n, job in enumerate(as_completed(jobs), 1):
            row = job.result()
            tic = row["tic"]
            if row.get("level") in ("CANDIDATE", "REVIEW", "REJECT") and row.get("time"):
                row["known_as"] = known_variable(info[tic]["ra"], info[tic]["dec"])
            write_row(row)
            rows.append(row)
            counts[row.get("level", "?")] += 1
            detail = (f"{row['depth_ppm']} ppm, {row['duration_hours']} h, SNR {row['snr']}: "
                      if row.get("time") and row.get("level") != "NOTHING" else "")
            print(f"[{n}/{len(picks)}] TIC {tic}: {detail}{row.get('verdict')}", flush=True)
            if time.time() - last_note > 120 or n == len(picks):
                elapsed = (time.time() - t_search) / 60
                eta = elapsed / n * (len(picks) - n)
                print(f"   == progress: {n}/{len(picks)} stars, {elapsed:.1f} min so far, ~{eta:.0f} min left | "
                      + ", ".join(f"{k} {v}" for k, v in counts.most_common()), flush=True)
                last_note = time.time()
    search_minutes = (time.time() - t_search) / 60

    followups = {}
    to_follow = [r for r in rows if r.get("level") in ("CANDIDATE", "REVIEW")
                 or (FOLLOW_UP_BINARY_REJECTS and r.get("level") == "REJECT" and str(r.get("binary_like")) == "True")]
    if FOLLOW_UP and to_follow:
        import step4_followup
        print(f"\nFollowing up {len(to_follow)} stars...")
        for i, r in enumerate(to_follow, 1):
            t0 = time.time()
            print(f"  [{i}/{len(to_follow)}] TIC {r['tic']} ({r['level']})...", flush=True)
            try:
                followups[r["tic"]] = step4_followup.follow_up(r, info.get(r["tic"], {}))
            except Exception as e:
                followups[r["tic"]] = f"follow-up failed: {e.__class__.__name__}: {e}"
            print(f"      done in {(time.time() - t0) / 60:.1f} min", flush=True)

    minutes = (time.time() - t_start) / 60
    report = write_report(rows, followups, minutes, removed)
    print("\n" + "=" * 70)
    print(f"{len(rows)} stars: search {search_minutes:.1f} min, total {minutes:.1f} min.")
    print("  " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    print(f"Report: {report}")
    print(f"Full log: {LOG}")


if __name__ == "__main__":
    main()
