"""
Tests for step 4 (new three-level version), compared with the old version (old/step4_old.py).

    python test_step4.py regression   known false alarms must be REJECTed, the binary flagged
    python test_step4.py planets      known long-period planets must come out CANDIDATE or REVIEW
    python test_step4.py injection    fake transits in quiet real stars: what do we keep, what do we lose?
    python test_step4.py all          everything (~15-20 min)

Results go to the "reports" folder (injection_recovery.md / .png / .csv).
"""
import csv
import random
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

import step4_single_transits as s4
from old import step4_old as old

warnings.filterwarnings("ignore")

LK_CACHE = Path.home() / ".cache" / "lightkurve" / "mastDownload" / "TESS"
REPORTS = s4.REPORTS

REGRESSION = [   # (TIC, expected, why)
    (356602976, "REJECT", "dip only appears with one cleaning setting"),
    (399912515, "REJECT", "background light jumps during the dip"),
    (451788144, "REJECT", "HIP 93933: thruster firing inside the dip"),
    (279916080, "REJECT", "10-hour pulsator"),
    (41602482, "BINARY", "TIC 41602470's eclipses leaking into the giant's light"),
    (229742371, "NOT CANDIDATE", "dip at the very start of the sector"),
    (259271084, "NOT CANDIDATE", "spotted star (VSX: ROT), lopsided dip"),
    (141533801, "REJECT", "flare: the light next to it looks like a dip"),
]
PLANETS = [   # (name, TIC, period d, transit mid-time BJD, duration h, how many orbits away to look)
    ("TOI-2180 b", 298663873, 260.79, 2458830.7652, 24.1, 0),    # period uncertain: only its own epoch
    ("TOI-2010 b", 26547036, 141.834025, 2458712.30168, 8.68, 8),
    ("TOI-791 c", 306472057, 232.0157, 2459433.98252, 11.93, 5),
]
REVIEW_STARS = [   # REVIEW in the 15 Nov 2026 run: where do they land now?
    (138131958, "weak 20-h dip mid-sector"),
    (362215490, "dip at ~day 2412, next to a spacecraft event"),
    (296079775, "dip at ~day 2412, next to a spacecraft event"),
    (198099774, "slow sag near the sector end"),
]
N_INJ_STARS = 50
DEPTHS_PPM = [300, 600, 1000, 2000, 3500, 5000]
DURATIONS_H = [2, 4, 8, 12, 16]
PER_BIN = 10                 # fake transits per depth x duration box
SEED = 7


# ======================================================================
# Helpers
# ======================================================================
@lru_cache(maxsize=64)
def load_cached(tic, sector=40):
    files = list(LK_CACHE.glob(f"*s{sector:04d}-{tic:016d}*/*lc.fits"))
    return s4.from_fits(files[0]) if files else None


def old_level(verdict):
    """Old version's verdicts on the new scale: PASSES ~ CANDIDATE, SUSPICIOUS ~ REVIEW."""
    if verdict.startswith("PASSES"):
        return "CANDIDATE"
    if verdict.startswith("SUSPICIOUS"):
        return "REVIEW"
    return "NOTHING"


def run_both(data, tc=None, D=None):
    """Run the new and old pipeline. Returns {"new": (level, found_here, details), "old": ...}."""
    out = {}
    res, _ = s4.search(data)
    here = tc is None or ("time" in res and abs(res["time"] - tc) < 0.5 * max(D, res["duration_hours"] / 24) + 1 / 24)
    out["new"] = (res["level"], here, res)
    ores, _ = old.search(data["t"], data["f"], data["radius"])
    ores = ores or {"verdict": "nothing found"}
    here_o = tc is None or ("time" in ores and abs(ores["time"] - tc) < 0.5 * max(D, ores["duration_hours"] / 24) + 1 / 24)
    out["old"] = (old_level(ores["verdict"]), here_o, ores)
    return out


def observable(t, tc, D):
    """Enough data during the transit to see it? (at least half the expected 2-minute points)"""
    return np.sum(np.abs(t - tc) < D / 2) >= 0.5 * D / (2 / 1440)


# ======================================================================
# 1. Regression stars
# ======================================================================
def test_regression():
    print("\n1. Regression stars (sector 40)")
    ok = 0
    for tic, expected, why in REGRESSION:
        data = load_cached(tic) or s4.download(tic, 40)
        res, _ = s4.search(data)
        level = res["level"]
        if expected == "BINARY":
            good = level == "REJECT" and bool(res.get("binary_like"))
        elif expected == "NOT CANDIDATE":
            good = level != "CANDIDATE"
        else:   # "REJECT": rejected, or not even detected (NOTHING) - either way never shown as a candidate
            good = level in ("REJECT", "NOTHING")
        ok += good
        fails = "; ".join(filter(None, [res.get("hard_fails"), res.get("soft_fails")]))
        print(f"   {'OK  ' if good else 'FAIL'} TIC {tic} ({why}): expected {expected}, got {level}"
              + (f" [binary-like]" if res.get("binary_like") else ""))
        if "time" in res:
            print(f"        event: day {res['time']}, {res['duration_hours']} h, {res['depth_ppm']} ppm, "
                  f"SNR {res['snr']}, SAP ratio {res.get('sap_ratio')}, background ratio {res.get('bkg_ratio')}")
        if fails:
            print(f"        {fails[:300]}")
    print(f"   {ok} of {len(REGRESSION)} as expected")
    print("\n   REVIEW stars from the 15 Nov run, with today's checks:")
    for tic, why in REVIEW_STARS:
        res, _ = s4.search(load_cached(tic) or s4.download(tic, 40))
        print(f"   TIC {tic} ({why}): {res['level']}"
              + (f", day {res['time']}, {res['duration_hours']} h, {res['depth_ppm']} ppm, common-mode fraction "
                 f"{res.get('common_frac')}" if "time" in res else ""))
        fails = "; ".join(filter(None, [res.get("hard_fails"), res.get("soft_fails")]))
        if fails:
            print(f"        {fails[:300]}")
    return ok


# ======================================================================
# 2. Known long-period planets
# ======================================================================
def transit_sectors(tic, period, tmid, dur_h, k_max):
    """Sectors with 2-min data that contain a transit, from MAST's observation times (no downloads)."""
    from astroquery.mast import Observations
    obs = Observations.query_criteria(obs_collection="TESS", dataproduct_type="timeseries",
                                      target_name=str(tic), provenance_name="SPOC")
    found = []
    for o in obs:
        if o["t_max"] - o["t_min"] > 40:          # multi-sector combined products: skip
            continue
        start, end = o["t_min"] + 2400000.5, o["t_max"] + 2400000.5      # MJD -> JD
        for k in range(-k_max, k_max + 1):
            tc = tmid + k * period
            if start + 0.5 < tc < end - 0.5:
                found.append((int(o["sequence_number"]), tc - 2457000))     # JD -> TESS days (BTJD)
    return sorted(set(found))


def test_planets():
    print("\n2. Known long-period planets (must be CANDIDATE or REVIEW)")
    rows = []
    for name, tic, period, tmid, dur_h, k_max in PLANETS:
        sectors = transit_sectors(tic, period, tmid, dur_h, k_max)
        if not sectors:
            print(f"   {name}: no 2-minute sector contains a transit")
            continue
        for sector, tc in sectors:
            data = s4.download(tic, sector)
            if data is None:
                print(f"   {name}: sector {sector} download gave nothing")
                continue
            D = dur_h / 24
            if not observable(data["t"], tc, D):
                print(f"   {name}, sector {sector}: transit at day {tc:.2f} falls in a data gap (not observable)")
                continue
            both = run_both(data, tc, D)
            for which in ("new", "old"):
                level, here, res = both[which]
                kept = here and level in ("CANDIDATE", "REVIEW")
                rows.append((name, sector, which, level if here else "LOST", kept))
                fails = "; ".join(filter(None, [res.get("hard_fails"), res.get("soft_fails")])) if which == "new" else res.get("verdict", "")
                print(f"   {'OK  ' if kept else 'MISS'} {name}, sector {sector}, transit day {tc:.2f} ({dur_h:.1f} h), "
                      f"{which}: {level if here else 'not found at the transit'}"
                      + (f" (found {res.get('duration_hours')} h, {res.get('depth_ppm')} ppm, SNR {res.get('snr')})" if here and 'snr' in res else ""))
                if fails and which == "new":
                    print(f"        {fails[:300]}")
    return rows


# ======================================================================
# 3. Injection-recovery
# ======================================================================
def quiet_stars():
    """Stars from the log with 'nothing found', a saved light curve, and not too wiggly."""
    rows = [r for r in s4.read_log() if (r.get("verdict") or "").startswith("nothing found")
            and r.get("level") in (None, "", "NOTHING") and r.get("sector") in ("40", 40)]
    tics = sorted({int(r["tic"]) for r in rows})
    good = []
    for tic in tics:
        data = load_cached(tic)
        if data is None or data["t"].size < 1000:
            continue
        flat = s4.hunter.detrend(data["t"], data["f"], np.zeros(data["t"].size, bool), window=s4.MIN_CLEAN_WINDOW)
        if s4.wiggliness(data["t"], flat) <= s4.MAX_WIGGLE:
            good.append(tic)
    random.Random(SEED).shuffle(good)
    return good[:N_INJ_STARS]


def transit_shape(t, tc, D, depth):
    """Flat bottom, short straight slopes in and out (12% of the duration each)."""
    return 1 - depth * np.clip((D / 2 - np.abs(t - tc)) / (0.12 * D), 0, 1)


def inject_one(task):
    """Runs in a helper process: one fake transit through both pipelines."""
    tic, depth_ppm, dur_h, tc = task
    data = load_cached(tic)
    D, depth = dur_h / 24, depth_ppm / 1e6
    base = {"tic": tic, "depth_ppm": depth_ppm, "dur_h": dur_h, "tc": round(tc, 3)}
    if not observable(data["t"], tc, D):
        return dict(base, observable=False)
    fake = dict(data)
    shape = transit_shape(data["t"], tc, D, depth)
    fake["f"], fake["sap"] = data["f"] * shape, data["sap"] * shape     # background and flags unchanged
    try:
        both = run_both(fake, tc, D)
    except Exception as e:
        return dict(base, observable=True, error=f"{e.__class__.__name__}: {e}")
    out = dict(base, observable=True)
    for which in ("new", "old"):
        level, here, res = both[which]
        if level == "NOTHING" or (level == "REJECT" and "too variable" in (res.get("hard_fails") or "")):
            outcome = level                  # nothing passed the threshold (or star skipped): too weak
        elif not here:
            outcome = "OTHER DIP"            # a different dip, above the threshold, was stronger
        else:
            outcome = level
        out[f"{which}_outcome"] = outcome
        out[f"{which}_hard"] = res.get("hard_fails", "") if which == "new" else ""
        out[f"{which}_soft"] = res.get("soft_fails", "") if which == "new" else ""
        out[f"{which}_verdict"] = res.get("verdict", "")
    return out


def summarise(results, which):
    obs = [r for r in results if r.get("observable") and not r.get("error")]
    lines = [f"### {which.upper()} pipeline", "",
             "Each box: % CANDIDATE / % CANDIDATE+REVIEW  (n = observable fake transits)", "",
             "| duration \\ depth | " + " | ".join(f"{d} ppm" for d in DEPTHS_PPM) + " |",
             "|---|" + "---|" * len(DEPTHS_PPM)]
    grid_c = np.full((len(DURATIONS_H), len(DEPTHS_PPM)), np.nan)
    grid_cr = grid_c.copy()
    for i, dur in enumerate(DURATIONS_H):
        cells = []
        for j, dep in enumerate(DEPTHS_PPM):
            sel = [r for r in obs if r["dur_h"] == dur and r["depth_ppm"] == dep]
            if not sel:
                cells.append("-")
                continue
            c = np.mean([r[f"{which}_outcome"] == "CANDIDATE" for r in sel])
            cr = np.mean([r[f"{which}_outcome"] in ("CANDIDATE", "REVIEW") for r in sel])
            grid_c[i, j], grid_cr[i, j] = c, cr
            cells.append(f"{c * 100:.0f} / {cr * 100:.0f} (n={len(sel)})")
        lines.append(f"| {dur} h | " + " | ".join(cells) + " |")
    n = len(obs)
    tot = Counter(r[f"{which}_outcome"] for r in obs)
    lines += ["", f"Overall ({n} observable): " + ", ".join(
        f"{k} {v} ({v / n * 100:.0f}%)" for k, v in tot.most_common()), ""]
    return lines, grid_c, grid_cr


def killers(results):
    """Which checks removed fake transits (new pipeline), and which put them in REVIEW."""
    obs = [r for r in results if r.get("observable") and not r.get("error")]
    lost, review = Counter(), Counter()
    for r in obs:
        out = r["new_outcome"]
        if out == "REJECT":
            for x in filter(None, r["new_hard"].split("; ")):
                lost[s4.check_name(x)] += 1
        elif out == "NOTHING":
            lost["too weak: nothing found"] += 1
        elif out == "OTHER DIP":
            lost["a different dip was stronger"] += 1
        elif out == "REVIEW":
            for x in filter(None, r["new_soft"].split("; ")):
                review[s4.check_name(x)] += 1
    return lost, review


def test_injection():
    print("\n3. Injection-recovery")
    stars = quiet_stars()
    print(f"   {len(stars)} quiet stars; {len(DEPTHS_PPM) * len(DURATIONS_H) * PER_BIN} fake transits")
    rng = random.Random(SEED)
    combos = [(dep, dur) for dep in DEPTHS_PPM for dur in DURATIONS_H for _ in range(PER_BIN)]
    rng.shuffle(combos)
    tasks = []
    for i, (dep, dur) in enumerate(combos):
        tic = stars[i % len(stars)]
        t = load_cached(tic)["t"]
        tasks.append((tic, dep, dur, rng.uniform(t.min() + 0.3, t.max() - 0.3)))
    t0, results = time.time(), []
    with ProcessPoolExecutor(max_workers=s4.PARALLEL) as pool:
        for k, r in enumerate(pool.map(inject_one, tasks, chunksize=4), 1):
            results.append(r)
            if k % 50 == 0 or k == len(tasks):
                print(f"   {k}/{len(tasks)} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    REPORTS.mkdir(exist_ok=True)
    keys = sorted({k for r in results for k in r})
    with (REPORTS / "injection_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    n_all = len(results)
    n_obs = sum(bool(r.get("observable")) for r in results)
    errors = [r for r in results if r.get("error")]
    md = ["# Injection-recovery test", "",
          f"{n_all} fake transits in {len(stars)} quiet sector-40 stars; {n_all - n_obs} fell in data gaps "
          f"(not observable, left out); {len(errors)} errors.", ""]
    grids = {}
    for which in ("new", "old"):
        lines, gc, gcr = summarise(results, which)
        md += lines
        grids[which] = (gc, gcr)
    lost, review = killers(results)
    md += ["### What removed fake transits (new pipeline)", "", "| Reason | Fake transits |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in lost.most_common()]
    md += ["", "### Why fake transits ended up in REVIEW (new pipeline)", "", "| Soft check | Fake transits |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in review.most_common()]
    if errors:
        md += ["", "### Errors", ""] + [f"- {r['tic']}: {r['error']}" for r in errors[:10]]
    (REPORTS / "injection_recovery.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, which in enumerate(("old", "new")):
        for col, (grid, title) in enumerate(zip(grids[which], ("% CANDIDATE", "% CANDIDATE + REVIEW"))):
            ax = axes[row, col]
            im = ax.imshow(grid * 100, origin="lower", cmap="viridis", vmin=0, vmax=100, aspect="auto")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if np.isfinite(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j] * 100:.0f}", ha="center", va="center",
                                color="white" if grid[i, j] < 0.6 else "black", fontsize=9)
            ax.set_xticks(range(len(DEPTHS_PPM)), [str(d) for d in DEPTHS_PPM])
            ax.set_yticks(range(len(DURATIONS_H)), [f"{d} h" for d in DURATIONS_H])
            ax.set_xlabel("Depth (ppm)")
            ax.set_title(f"{which.upper()} pipeline: {title}")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Injection-recovery: fake single transits in quiet sector-40 stars")
    fig.tight_layout()
    fig.savefig(REPORTS / "injection_recovery.png", dpi=120)
    plt.close(fig)
    print(f"\n   Saved reports/injection_recovery.md, .png, injection_results.csv ({(time.time() - t0) / 60:.1f} min)")
    return results


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("regression", "all"):
        test_regression()
    if what in ("planets", "all"):
        test_planets()
    if what in ("injection", "all"):
        test_injection()
