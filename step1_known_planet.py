"""
Planet hunt, step 1: find a planet we already know is there.

Star:   WASP-18, which has a giant planet (bigger than Jupiter) orbiting every ~0.94 days.
Goal:   get its dips out of real TESS data and measure the planet ourselves.

What it does (one panel each):
    1. Downloads one TESS sector (~27 days) of brightness measurements.
    2. Removes the star's slow wobbles ("flattening" / detrending).
    3. BLS search: tries 20,000 possible periods and scores how dip-shaped each one is.
    4. Folds the data at the best period: every dip stacked on top of each other.
Then it prints the period, depth and an estimate of the planet's size.

Run:   python step1_known_planet.py   (needs internet the first time; takes a minute)
Needs: lightkurve   (pip install lightkurve)
"""
from pathlib import Path

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

STAR = "WASP-18"
KNOWN_PERIOD = 0.9415          # days, the published value, to compare with
SUN_IN_EARTHS = 109.1          # the Sun's radius in Earth radii
SUN_IN_JUPITERS = 9.73         # the Sun's radius in Jupiter radii


def main():
    print(f"Searching TESS data for {STAR}...")
    results = lk.search_lightcurve(STAR, mission="TESS", author="SPOC")
    if len(results) == 0:
        print("Nothing found. Check your internet connection and try again.")
        return
    print(f"Found {len(results)} sectors. Downloading the first one...")

    lc = results[0].download().remove_nans().normalize()
    flat = lc.flatten(window_length=901)      # 901 points x 2 min = ~30 hours

    print("Running the BLS search (tries 20,000 periods)...")
    periods = np.linspace(0.5, 10, 20000)
    bls = flat.to_periodogram(method="bls", period=periods, duration=[0.05, 0.08, 0.1, 0.15])
    period = bls.period_at_max_power.value
    t0 = bls.transit_time_at_max_power
    depth = float(np.asarray(bls.depth_at_max_power))

    folded = flat.fold(period=period, epoch_time=t0)

    # ---- the four panels ----
    fig, axes = plt.subplots(4, 1, figsize=(11, 13))
    lc.plot(ax=axes[0], color="gray")
    axes[0].set_title("1. Raw TESS light curve (brightness over ~27 days)")
    flat.plot(ax=axes[1], color="black")
    axes[1].set_title("2. After removing slow wobbles: each spike down is one transit")
    bls.plot(ax=axes[2])
    axes[2].axvline(period, color="red", alpha=0.4, lw=4)
    axes[2].set_title(f"3. BLS search: the tallest peak is the period ({period:.4f} days)")
    folded.scatter(ax=axes[3], color="lightgray", s=2, label="every measurement")
    folded.bin(time_bin_size=0.005).scatter(ax=axes[3], color="red", s=8, label="averaged")
    axes[3].set_xlim(-0.25, 0.25)
    axes[3].set_title("4. Folded: all the dips stacked on top of each other")
    fig.tight_layout()

    out = Path(__file__).with_name(f"{STAR}_step1.png")
    fig.savefig(out, dpi=150)

    # ---- the numbers ----
    ratio = np.sqrt(depth)                     # planet radius / star radius
    star_radius = lc.meta.get("RADIUS")        # in Sun radii, from the TESS catalogue
    print("\n" + "=" * 50)
    print(f"Period:     {period:.4f} days   (published: {KNOWN_PERIOD} days)")
    print(f"Depth:      {depth * 100:.2f} %  -> planet radius = {ratio:.3f} x star radius")
    if star_radius:
        print(f"Star:       {star_radius:.2f} x the Sun's radius")
        print(f"Planet:     {ratio * star_radius * SUN_IN_JUPITERS:.2f} Jupiters "
              f"= {ratio * star_radius * SUN_IN_EARTHS:.1f} Earths")
    print("=" * 50)
    print(f"Graph saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
