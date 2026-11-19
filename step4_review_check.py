"""
Planet hunt: a closer look at the REVIEW stars from a step-4 run.

    1. Same-time test: stack ~20-30 other stars from the same camera/CCD (and from all
       cameras) around each event, in cleaned (PDCSAP) and raw (SAP) flux. If they dip
       together, it's the spacecraft. Compared with 60 random moments, so we know how
       often such a "common dip" happens by chance.
    2. Pixel test (difference image, from step 5): is the lost light centred on the star?
    3. All TESS data, including full-frame-image light curves (TESS-SPOC 200/600/1800 s,
       QLP, TGLC, eleanor): a second dip of the same depth AND length? Which periods are
       still allowed by physics and by all the coverage?
    4. A suggested verdict (you decide in the end).

Run:   python step4_review_check.py        (edit STARS below)
Writes reports/review_check.md and plots in reports/review_check/.
"""
import contextlib
import io
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import step4_followup as fu
import step4_single_transits as s4
import step5_followup as f5

warnings.filterwarnings("ignore")

SECTOR = 40
STARS = [   # (TIC, event day, duration h, depth ppm) from the 15 Nov 2026 run
    (138131958, 2401.576, 20.0, 773),
    (362215490, 2411.854, 12.0, 375),
    (296079775, 2411.606, 24.0, 147),
    (198099774, 2417.286, 20.0, 1068),
]
FULL_FOLLOW_UP = {138131958}          # stars that also get the all-sectors search (slow)
N_RANDOM = 60
OUT = s4.REPORTS / "review_check"


def load(tic):
    files = list(s4.LK_CACHE.glob(f"*s{SECTOR:04d}-{tic:016d}*/*lc.fits"))
    return s4.from_fits(files[0]) if files else s4.download(tic, SECTOR)


def stack(refs, tc, D, column):
    """Dip measured in every reference star at the same moment. column 2 = cleaned, 3 = raw SAP."""
    d, e = [], []
    for r in refs:
        dd, ee = s4.binned_dip(r[1].astype(float), r[column].astype(float), tc, D)
        if np.isfinite(dd) and ee > 0:
            d.append(dd)
            e.append(ee)
    d, e = np.array(d), np.array(e)
    if d.size < 5:
        return None
    err = 1.2533 * s4.robust_noise(d) / np.sqrt(d.size)        # error of the median
    return {"n": int(d.size), "frac": float(np.mean(d / e > 3)), "median": float(np.median(d)),
            "z": float(np.median(d) / err) if err > 0 else 0.0}


def residual(tb, fb, tc, D):
    """Flux around the event relative to a straight line fitted beside the dip."""
    dt = tb - tc
    win = np.abs(dt) < 3 * D
    out = win & (np.abs(dt) > 0.8 * D)
    if out.sum() < 5:
        return None, None
    line = np.polyval(np.polyfit(dt[out], fb[out], 1), dt[win])
    return dt[win] * 24, fb[win] / line - 1


def same_time(tic, tc, D, depth, data):
    ref = s4.build_reference(SECTOR, verbose=False)
    same = [r for r in ref["groups"].get((data["camera"], data["ccd"]), []) if r[0] != tic]
    every = [r for g in ref["groups"].values() for r in g if r[0] != tic]
    controls = np.random.default_rng(tic % 1000).uniform(data["t"].min() + 1, data["t"].max() - 1, N_RANDOM)
    lines, result = [], {}
    for label, refs in ((f"same camera/CCD ({data['camera']}/{data['ccd']})", same), ("all cameras", every)):
        for column, name in ((2, "cleaned"), (3, "raw SAP")):
            ev = stack(refs, tc, D, column)
            if ev is None:
                continue
            ctrl = [c for c in (stack(refs, c, D, column) for c in controls) if c]
            pct = float(np.mean([c["z"] < ev["z"] for c in ctrl]) * 100)
            ratio = ev["median"] / depth
            result[(label.split(" (")[0], name)] = dict(ev, pct=pct, ratio=ratio,
                                                        ctrl_frac=float(np.median([c["frac"] for c in ctrl])))
            lines.append(f"{label}, {name} flux, {ev['n']} stars: {ev['frac'] * 100:.0f}% dip by > 3 sigma "
                         f"(random moments: median {np.median([c['frac'] for c in ctrl]) * 100:.0f}%); stacked median "
                         f"{ev['median'] * 1e6:+.0f} ppm = {ratio * 100:.0f}% of the target's dip, {ev['z']:.1f} sigma, "
                         f"stronger than at {pct:.0f}% of {len(ctrl)} random moments")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, column, name, own in ((axes[0], 2, "cleaned (PDCSAP)", data["f"]), (axes[1], 3, "raw (SAP)", data["sap"])):
        curves = []
        for r in same:
            x, y = residual(r[1].astype(float), r[column].astype(float), tc, D)
            if x is not None:
                ax.plot(x, y * 1e6, color="gray", lw=0.4, alpha=0.4)
                curves.append((x, y))
        if curves:
            grid = np.arange(-3 * D * 24, 3 * D * 24, 0.5)
            med = np.nanmedian([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves], axis=0)
            ax.plot(grid, med * 1e6, color="black", lw=2, label=f"median of {len(curves)} stars, same CCD")
        tb, fb = s4.hunter.bin_data(data["t"], own, size=s4.BIN)
        x, y = residual(tb, fb, tc, D)
        if x is not None:
            ax.plot(x, y * 1e6, "o-", color="red", ms=3, lw=1, label=f"TIC {tic}")
        ax.axvspan(-D / 2 * 24, D / 2 * 24, color="orange", alpha=0.15)
        ax.set_title(f"{name} flux", fontsize=10)
        ax.set_xlabel(f"hours from day {tc}")
        ax.set_ylim(-4 * depth * 1e6, 3 * depth * 1e6)
    axes[0].set_ylabel("brightness change (ppm)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"TIC {tic}: do other stars on camera {data['camera']} CCD {data['ccd']} dip at the same time?")
    fig.tight_layout()
    out = OUT / f"TIC_{tic}_same_time.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return lines, result, out


def pixel(tic, tc, D, depth):
    info = f5.tic_info(tic)
    buf = io.StringIO()
    out = OUT / f"TIC_{tic}_pixels.png"
    with contextlib.redirect_stdout(buf):
        f5.pixel_test(tic, info, lambda t: np.abs(t - tc) < 0.4 * D,
                      lambda t: (np.abs(t - tc) > 0.8 * D) & (np.abs(t - tc) < 2.2 * D),
                      depth, f"TIC {tic}: where is the light lost during the dip at day {tc}?", out)
    text = buf.getvalue()
    snr = re.search(r"Strongest light-loss pixel: SNR ([\d.]+)", text)
    off = re.search(r"centred ([\d.]+)\" \(([\d.]+) pixels\)", text)
    target_snr = re.search(rf"{tic}\s+[\d.]+\s+0\.0\"\s+[\d.]+\"\s+(-?[\d.]+)", text)
    summary = (f"strongest light-loss pixel SNR {snr.group(1) if snr else '?'}; centre {off.group(1) if off else '?'}\" "
               f"({off.group(2) if off else '?'} px) from the star; SNR at the star's own pixel "
               f"{target_snr.group(1) if target_snr else '?'}")
    star_snr = float(target_snr.group(1)) if target_snr else 0.0
    return summary, (float(snr.group(1)) if snr else 0.0), (float(off.group(2)) if off else 0.0), star_snr, out


def verdict(st, pix_snr, pix_off, star_snr):
    """
    Spacecraft: at least half the reference stars dip (and at least twice as many as at random
    moments), and their median dip is at least half the target's dip, so it could explain it.
    Neighbour: the lost light is centred elsewhere AND the star's own pixel doesn't lose light.
    """
    reasons = []
    for (scope, flux), m in st.items():
        if m["frac"] >= 0.5 and m["frac"] >= 2 * m["ctrl_frac"] and m["ratio"] >= 0.5:
            reasons.append(f"{scope}, {flux} flux: {m['frac'] * 100:.0f}% of {m['n']} stars dip "
                           f"(random moments {m['ctrl_frac'] * 100:.0f}%), median glitch {m['ratio'] * 100:.0f}% of the dip")
    if reasons:
        return "spacecraft (other stars dip at the same time)", reasons
    if pix_snr >= 5 and pix_off > 1.0 and star_snr < 1.5:
        return "probably a neighbouring star", [f"lost light centred {pix_off:.1f} px away, star's own pixel SNR {star_snr:.1f}"]
    if pix_snr >= 5 and pix_off > 1.0 and star_snr < 3:
        return "not the spacecraft; pixel test unclear (some light lost at the star, more nearby)", [
            f"lost light centred {pix_off:.1f} px away, star's own pixel SNR {star_snr:.1f}"]
    return "not explained by the spacecraft or a neighbour (star variability, or still possible)", []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# REVIEW stars: closer look", ""]
    table = ["| TIC | event | same-CCD cleaned stack | same-CCD raw stack | pixel test | suggested verdict |",
             "|---|---|---|---|---|---|"]
    for tic, tc, dur_h, depth_ppm in STARS:
        print(f"\n=== TIC {tic}: day {tc}, {dur_h} h, {depth_ppm} ppm ===", flush=True)
        D, depth = dur_h / 24, depth_ppm / 1e6
        data = load(tic)
        lines, st, plot = same_time(tic, tc, D, depth, data)
        for line in lines:
            print("  " + line)
        pix_text, pix_snr, pix_off, star_snr, pix_plot = pixel(tic, tc, D, depth)
        print("  pixel test: " + pix_text)
        follow = []
        if tic in FULL_FOLLOW_UP:
            print("  all TESS sectors (incl. full-frame images)...", flush=True)
            follow = fu.second_dip(tic, SECTOR, tc, D, depth, fu.star_info(tic, {}))
            for line in follow:
                print("    " + line)
        v, why = verdict(st, pix_snr, pix_off, star_snr)
        print(f"  suggested verdict: {v}" + (f" ({'; '.join(why)})" if why else ""))

        def cell(key):
            m = st.get(key)
            return f"{m['ratio'] * 100:+.0f}% of dip, {m['z']:.1f}σ, >{m['pct']:.0f}% of random" if m else "-"

        table.append(f"| {tic} | day {tc}, {dur_h:.0f} h, {depth_ppm} ppm | {cell(('same camera/CCD', 'cleaned'))} | "
                     f"{cell(('same camera/CCD', 'raw SAP'))} | {pix_text} | {v} |")
        md += [f"## TIC {tic}", "", f"Event: day {tc}, {dur_h} h, {depth_ppm} ppm (camera {data['camera']}, "
               f"CCD {data['ccd']})", "", "**Same-time test**", ""] + [f"- {x}" for x in lines]
        md += ["", f"![same time](review_check/{plot.name})", "", f"**Pixel test**: {pix_text}", "",
               f"![pixels](review_check/{pix_plot.name})", ""]
        if follow:
            md += ["**All TESS data**", ""] + [f"- {x}" for x in follow] + [""]
        md += [f"**Suggested verdict**: {v}" + (f" ({'; '.join(why)})" if why else ""), ""]
    md[2:2] = table + [""]
    (s4.REPORTS / "review_check.md").write_text("\n".join(md), encoding="utf-8")
    print("\nWrote reports/review_check.md")


if __name__ == "__main__":
    main()
