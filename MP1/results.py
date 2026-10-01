"""
Extract the oscillation period from each Monte Carlo .step run in an LTspice
.raw file, and plot a histogram of the results.

Requires:
    pip install PyLTSpice matplotlib numpy

Usage:
    python3 period_histogram.py MP1.raw
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PyLTSpice import RawRead


def find_period(t, v, threshold):
    """
    Estimate the oscillation period from a time-domain trace by finding
    rising-edge threshold crossings (with linear interpolation for
    sub-timestep accuracy) and averaging the spacing between them.
    """
    # Boolean array: is the sample above the threshold?
    above = v > threshold

    # Rising edges = transitions from False -> True
    rising_idx = np.where(np.diff(above.astype(int)) == 1)[0]

    if len(rising_idx) < 2:
        return None  # not enough crossings to measure a period

    # Linear interpolation to get a more precise crossing time for each edge
    crossing_times = []
    for i in rising_idx:
        t0, t1 = t[i], t[i + 1]
        v0, v1 = v[i], v[i + 1]
        # avoid div by zero on a flat/duplicate step
        if v1 == v0:
            continue
        frac = (threshold - v0) / (v1 - v0)
        crossing_times.append(t0 + frac * (t1 - t0))

    if len(crossing_times) < 2:
        return None

    crossing_times = np.array(crossing_times)
    periods = np.diff(crossing_times)

    # Use the median period across all cycles in this run (robust to any
    # single noisy/short first or last partial cycle)
    return np.median(periods)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 period_histogram.py <path_to_MP1.raw>")
        sys.exit(1)

    raw_path = sys.argv[1]

    print(f"Loading {raw_path} ...")
    raw = RawRead(raw_path)

    # Figure out how many .step runs are in the file
    steps = raw.get_steps()
    n_steps = len(steps) if steps is not None else 1
    print(f"Found {n_steps} step(s).")

    vout_name = "V(vout)"  # LTspice raw files are case-insensitive on names
    time_name = "time"

    # Midpoint of the rail-to-rail swing (Vdd=3.3V referenced to 0V here) is
    # a reasonable default threshold for a rail-to-rail oscillator. Adjust
    # if your oscillator swings between different levels.
    threshold = 1.65

    periods = []

    for step_idx in range(n_steps):
        t = np.array(raw.get_trace(time_name).get_wave(step_idx))
        v = np.array(raw.get_trace(vout_name).get_wave(step_idx))

        period = find_period(t, v, threshold)
        if period is not None:
            periods.append(period)
        else:
            print(f"  step {step_idx}: could not find enough edges, skipping")

    periods = np.array(periods)
    print(f"\nMeasured period on {len(periods)} / {n_steps} runs.")
    print(f"Mean period:   {periods.mean()*1e3:.4f} ms")
    print(f"Std dev:       {periods.std()*1e3:.4f} ms")
    print(f"Min / Max:     {periods.min()*1e3:.4f} / {periods.max()*1e3:.4f} ms")

    # --- Plot histogram ---
    plt.figure(figsize=(8, 5))
    plt.hist(periods * 1e3, bins=30, edgecolor="black")
    plt.xlabel("Period (ms)")
    plt.ylabel("Count")
    plt.title(f"Monte Carlo Oscillation Period Distribution (n={len(periods)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("period_histogram.png", dpi=150)
    print("\nSaved plot to period_histogram.png")
    plt.show()


if __name__ == "__main__":
    main()