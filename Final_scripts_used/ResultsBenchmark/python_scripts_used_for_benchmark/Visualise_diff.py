#!/usr/bin/env python3
"""
======================================================================
Annual Kelvin “Winner” Plot  ─ Run-to-Run Comparison
======================================================================

• Reads two *.jsonl* prediction files.
• Uses only the first `MAX_LINES` rows from each file.
• Extracts the **year** from the `"id"` field (expects a 4-digit YYYY).
• Extracts the predicted temperature from `"prediction_raw_text"`:
      Kelvin directly, or °C converted to Kelvin.
• Computes the average Kelvin for each year in each run.
• Plots the signed difference  (Run 2 − Run 1) as a bar chart:
      – Bars > 0  ⇒ Run 2 warmer  (colour `COL_RUN2`)
      – Bars < 0  ⇒ Run 1 warmer  (colour `COL_RUN1`)
• Legend shows which colour means which run is warmer.
• Upper-right textbox states “N of M years Run 2 warmer”.
• Optionally saves the finished plot to `OUTPUT_PLOT`.

──────────────────────────────────────────────────────────────────────
Edit the CONFIGURATION block below, then run:
    python plot_kelvin_winner.py
"""

# ── CONFIGURATION ───────────────────────────────────────────────────
FILE1       = r"/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/Normal_data/infer_7B_bs4_interval1_allYears_compiled_12431721.jsonl"
FILE2       = r"/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/1pctco2_data/infer_7B_bs4_interval1_allYears_compiled_12431794.jsonl"

LABEL1      = "Run 12431721"
LABEL2      = "Run 12431794"

MAX_LINES   = 1_800              # first N rows from each file

COL_RUN1    = "#F7B513"          # gold / yellow
COL_RUN2    = "#E24329"          # red / orange

# Where to save the figure (PNG / PDF / SVG …).  Leave empty ("") to skip.
OUTPUT_PLOT = r"/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/NON_labeled_diff.png"
# ───────────────────────────────────────────────────────────────────


# — Standard lib
import json
import re
from collections import defaultdict, OrderedDict
from pathlib import Path

# — Third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter


# ═══════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════
ID_YEAR_RX  = re.compile(r"(\d{4})")                       # grabs YYYY
RX_KELVIN   = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:k|kelvin)\b", re.I)
RX_CELSIUS  = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°?\s*c|celsius)\b", re.I)

def celsius_to_kelvin(c: float) -> float:
    """Convert °C to K."""
    return c + 273.15


def extract_kelvin(text: str) -> float | None:
    """
    Pick the first temperature value in *text*.
    • ‘xxx K’ → Kelvin directly
    • ‘xxx °C’ → convert to Kelvin
    • Bare number → treat as Kelvin if 150 ≤ K ≤ 400
    """
    if not isinstance(text, str):
        return None

    m = RX_KELVIN.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None

    m = RX_CELSIUS.search(text)
    if m:
        try:
            return celsius_to_kelvin(float(m.group(1)))
        except ValueError:
            return None

    # Fallback: any number that looks like a plausible Kelvin
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if m:
        try:
            v = float(m.group(1))
            if 150 <= v <= 400:
                return v
        except ValueError:
            pass
    return None


def yearly_averages(path: str, max_lines: int) -> OrderedDict[int, float]:
    """
    Read up to *max_lines* JSONL rows, group Kelvin values by year,
    return {year → average Kelvin} sorted by year.
    """
    yearly: defaultdict[int, list[float]] = defaultdict(list)

    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            if ln > max_lines:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            m = ID_YEAR_RX.search(obj.get("id", ""))
            if not m:
                continue
            year = int(m.group(1))

            kelv = extract_kelvin(obj.get("prediction_raw_text", ""))
            if kelv is not None:
                yearly[year].append(kelv)

    return OrderedDict(
        (yr, sum(vals) / len(vals)) for yr, vals in sorted(yearly.items())
    )


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    avg1 = yearly_averages(FILE1, MAX_LINES)
    avg2 = yearly_averages(FILE2, MAX_LINES)

    if not avg1 or not avg2:
        print("⚠️  No data extracted – check file paths and JSON format.")
        return

    years   = sorted(set(avg1) & set(avg2))
    diffs   = np.array([avg2[y] - avg1[y] for y in years], dtype=float)
    colours = [COL_RUN2 if d > 0 else COL_RUN1 if d < 0 else "grey" for d in diffs]
    run2_wins = int(np.sum(diffs > 0))

    # ── Decide whether to switch to a symlog scale ──────────────────
    linthresh = np.percentile(np.abs(diffs), 90)     # linear region size
    use_symlog = np.max(np.abs(diffs)) > 3 * linthresh   # heuristic

    # ── Plot ─────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.bar(years, diffs, color=colours, width=0.8)
    ax.axhline(0, color="black", linewidth=0.7)

    if use_symlog:
        ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.text(
            0.02, 0.95,
            f"Sym-log scale (±{linthresh:.1f} K linear)",
            ha="left", va="top", transform=ax.transAxes,
            fontsize="small", color="dimgray"
        )

    ax.set_title(f"Annual winner by higher Kelvin (first {MAX_LINES:,} lines)")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Δ Temperature (K) ({LABEL2} – {LABEL1})")

    patch_run2 = mpatches.Patch(color=COL_RUN2, label=f"{LABEL2} higher")
    patch_run1 = mpatches.Patch(color=COL_RUN1, label=f"{LABEL1} higher")
    ax.legend(handles=[patch_run2, patch_run1], title="Kelvin winner")

    ax.text(
        0.98, 0.95,
        f"{run2_wins} of {len(years)} years\n{LABEL2} warmer",
        ha="right", va="top", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=0.5)
    )

    plt.tight_layout()

    # ── Save figure if requested ─────────────────────────────────────
    if OUTPUT_PLOT:
        out_path = Path(OUTPUT_PLOT).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"📈  Plot saved to {out_path}")
    else:
        print("ℹ️  OUTPUT_PLOT empty – figure not saved.")

    plt.show()


if __name__ == "__main__":
    main()