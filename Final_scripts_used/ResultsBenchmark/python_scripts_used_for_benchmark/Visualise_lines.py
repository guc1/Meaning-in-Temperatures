#!/usr/bin/env python3
"""
==================================================================
Year-by-Year Average Temperature Plot (first *N* lines per file)
==================================================================

• Reads two *.jsonl* prediction files.
• Uses only the first `MAX_LINES` rows from each file.
• Extracts the **year** from the `"id"` field (expects YYYY somewhere in it).
• Extracts the predicted temperature from `"prediction_raw_text"`,
  handling Kelvin directly or converting °C to K.
• Computes the average Kelvin value for each calendar year.
• Plots both series on a single Matplotlib figure with explicit colours.
• Optionally saves the figure to disk (PNG / PDF / SVG …).

──────────────────────────────────────────────────────────────────
Edit the CONFIGURATION block below, then run:
    python plot_yearly_avg_temps.py
"""

# ── CONFIGURATION ──────────────────────────────────────────────────────────
FILE1       = r"/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/Normal_data/infer_7B_bs4_interval1_allYears_compiled_12431721.jsonl"
FILE2       = r"/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/1pctco2_data/infer_7B_bs4_interval1_allYears_compiled_12431794.jsonl"

LABEL1      = "Run 12431721"
LABEL2      = "Run 12431794"

MAX_LINES   = 1_800          # first N rows from each file

# Explicit colours (keep in sync with other plots if desired)
COL_RUN1    = "#F7B513"      # gold / yellow
COL_RUN2    = "#E24329"      # red / orange

# Where to save the figure.  Leave empty ("") to *not* save.
OUTPUT_PLOT = r"/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/NON_labeled.png"
# ───────────────────────────────────────────────────────────────────────────


# — Standard library
import json, re
from collections import defaultdict, OrderedDict
from pathlib import Path

# — Third-party
import numpy as np
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════
# Helper functions (unchanged)
# ══════════════════════════════════════════════════════════════════
ID_YEAR_RX  = re.compile(r"(\d{4})")
RX_KELVIN   = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:k|kelvin)\b", re.I)
RX_CELSIUS  = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°?\s*c|celsius)\b", re.I)

def celsius_to_kelvin(c: float) -> float: return c + 273.15

def extract_kelvin(text: str) -> float | None:
    if not isinstance(text, str): return None
    m = RX_KELVIN.search(text)
    if m:
        try: return float(m.group(1))
        except ValueError: pass
    m = RX_CELSIUS.search(text)
    if m:
        try: return celsius_to_kelvin(float(m.group(1)))
        except ValueError: pass
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if m:
        try:
            v = float(m.group(1))
            if 150 <= v <= 400: return v
        except ValueError: pass
    return None

def yearly_averages(path: str, max_lines: int) -> OrderedDict[int, float]:
    yearly: defaultdict[int, list[float]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            if ln > max_lines: break
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue
            m = ID_YEAR_RX.search(obj.get("id", ""))
            if not m: continue
            year = int(m.group(1))
            kelv = extract_kelvin(obj.get("prediction_raw_text", ""))
            if kelv is not None: yearly[year].append(kelv)
    return OrderedDict((yr, sum(v)/len(v)) for yr, v in sorted(yearly.items()))


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    avg1 = yearly_averages(FILE1, MAX_LINES)
    avg2 = yearly_averages(FILE2, MAX_LINES)

    if not avg1 or not avg2:
        print("⚠️  No data extracted – check file paths & JSONL format.")
        return

    # ── Combine values to compute clipping thresholds ──────────────
    merged_vals = np.array(list(avg1.values()) + list(avg2.values()), dtype=float)
    lo_clip, hi_clip = np.percentile(merged_vals, [0.5, 99.5])

    # ── Filter years whose averages fall inside the band ───────────
    years_filt, vals1_filt, vals2_filt = [], [], []
    clipped_any = False
    for yr in sorted(set(avg1) & set(avg2)):
        v1, v2 = avg1[yr], avg2[yr]
        if lo_clip <= v1 <= hi_clip and lo_clip <= v2 <= hi_clip:
            years_filt.append(yr)
            vals1_filt.append(v1)
            vals2_filt.append(v2)
        else:
            clipped_any = True

    if not years_filt:
        print("🚫 All points considered outliers – nothing to plot.")
        return

    # ── Plot ───────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(years_filt, vals1_filt, marker="o", color=COL_RUN1, label=LABEL1)
    plt.plot(years_filt, vals2_filt, marker="o", color=COL_RUN2, label=LABEL2)

    plt.title(f"Yearly average predicted temperature (first {MAX_LINES:,} lines)")
    plt.xlabel("Year")
    plt.ylabel("Temperature (K)")
    plt.grid(True, linestyle="--", linewidth=0.3, alpha=0.6)
    plt.legend()

    if clipped_any:
        plt.text(0.02, 0.95, "Note: extreme outlier(s) excluded",
                 ha="left", va="top", transform=plt.gca().transAxes,
                 fontsize="small", color="dimgray")

    plt.tight_layout()

    # ── Save / show ────────────────────────────────────────────────
    if OUTPUT_PLOT:
        out = Path(OUTPUT_PLOT).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300)
        print(f"📈  Plot saved to {out}")
    else:
        print("ℹ️  OUTPUT_PLOT empty – figure not saved.")

    plt.show()


if __name__ == "__main__":
    main()