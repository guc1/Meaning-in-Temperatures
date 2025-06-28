#!/usr/bin/env python3
"""
Year-by-year points plot for the primary benchmark scenario
(uses the same extraction / scoring logic you had before, stripped down
to essentials).

Edit the CONFIGURATION block to change:
• points per sample
• level-penalty beta
• error cutoff
• month-sampling interval
• CLI override flag
• save-filtered-truth toggle
"""

# ── CONFIGURATION ─────────────────────────────────────────────────
POINTS_PER_SAMPLE      = 100      # editable
LEVEL_PENALTY          = 8        # editable (β)
ERROR_CUTOFF           = 5.0      # editable (K)
DEFAULT_MONTH_INTERVAL = 1        # editable
ALLOW_CLI_OVERRIDE     = True     # editable
SAVE_FILTERED_TRUTHS   = True     # editable
MONTH_INTERVAL         = 1  
POLY_DEGREE            = 10

# Default file paths (replace if needed)
TRUTH_FILE   = "/Users/yer/scriptie/Final_folder_scriptieV1/Data_Used_Model_Training_V2_ChatML/Test_data_sets_semantic/1pctCO2_increase_semantic/Non_labeled_test_1pctCO2_dataset/Non_labeled_test_1pctCO2_dataset.chatml.jsonl"
PRED_FILE    = "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/1pctco2_data/infer_7B_bs4_interval1_allYears_compiled_12431794.jsonl"
OUT_PLOT = "/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/1pctco2_data/yearly_points_Wrongly_1pctco2.png"
# ────────────────────────────────────────────────────────────────

import json, math, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ── identical regexes / helper from benchmark ───────────────────
RX_DATE      = re.compile(r"DATE=(\d{4})-(\d{2})")
_RX_KELVIN   = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:kelvin|k)\b", re.I)
_RX_CELSIUS  = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°?\s*c|celsius)\b", re.I)
_RX_PHRASE   = re.compile(
    r"(?i)(?:temperature[^0-9\-]{0,20}?is|value is|is|will be|will reach)[^0-9\-]{0,20}?(-?\d+(?:\.\d+)?)")
_RX_NUM      = re.compile(r"-?\d+(?:\.\d+)?")
PL_MIN_K, PL_MAX_K = 150.0, 400.0
PL_MIN_C, PL_MAX_C = -90.0, 70.0

def c2k(c: float) -> float: return c + 273.15

def _extract_value_from_text(text: str) -> float | None:
    """Faithful copy of the 'kelvin_and_celsius' branch from the benchmark."""
    if not isinstance(text, str): return None

    # explicit Kelvin
    for m in _RX_KELVIN.finditer(text):
        try: v = float(m.group(1))
        except ValueError: continue
        if PL_MIN_K <= v <= PL_MAX_K: return v

    # explicit Celsius
    for m in _RX_CELSIUS.finditer(text):
        try: v_c = float(m.group(1))
        except ValueError: continue
        if PL_MIN_C <= v_c <= PL_MAX_C:
            v_k = c2k(v_c)
            if PL_MIN_K <= v_k <= PL_MAX_K: return v_k

    # “temperature is 12” kind of phrases
    for m in _RX_PHRASE.finditer(text):
        try: v = float(m.group(1))
        except ValueError: continue
        if PL_MIN_C <= v <= PL_MAX_C: return c2k(v)
        if PL_MIN_K <= v <= PL_MAX_K: return v

    # bare number fallback
    nums = []
    for m in _RX_NUM.finditer(text):
        try: v = float(m.group(0))
        except ValueError: continue
        if PL_MIN_K <= v <= PL_MAX_K: nums.append(v)
        elif PL_MIN_C <= v <= PL_MAX_C: nums.append(c2k(v))
    if len(nums) == 1: return nums[0]
    return None

def log_penalty(err: float, beta: float, cutoff: float, pts: int) -> float:
    if math.isnan(err) or abs(err) >= cutoff: return 0.0
    up = math.log1p(beta * cutoff)
    return pts * (1 - math.log1p(beta * abs(err)) / up)

# ── STEP 1: read truth values + year tags ───────────────────────
truth_vals, years_by_line = [], {}
with Path(TRUTH_FILE).open(encoding="utf-8") as fh:
    for ln, line in enumerate(fh, 1):
        obj = json.loads(line)
        # grab DATE from first user message
        date_tag = None
        for msg in obj.get("messages", []):
            if msg.get("role") == "user":
                date_tag = RX_DATE.search(msg.get("content", "")); break
        if not date_tag: continue
        yr = int(date_tag.group(1)); mon = int(date_tag.group(2))
        if MONTH_INTERVAL > 1 and (mon - 1) % MONTH_INTERVAL: continue

        truth_k = _extract_value_from_text(json.dumps(obj))
        if truth_k is not None:
            truth_vals.append((ln, truth_k))
            years_by_line[ln] = yr

# ── STEP 2: read predictions line-by-line ───────────────────────
preds = {}
with Path(PRED_FILE).open(encoding="utf-8") as fh:
    for ln, line in enumerate(fh, 1):
        raw = json.loads(line).get("prediction_raw_text", "")
        preds[ln] = _extract_value_from_text(raw)

# ── STEP 3: score and accumulate per year ───────────────────────
sum_pts, count = defaultdict(float), defaultdict(int)
for ln, truth_k in truth_vals:
    pred_k = preds.get(ln)
    if pred_k is None: continue
    pts = log_penalty(abs(truth_k - pred_k), LEVEL_PENALTY, ERROR_CUTOFF, POINTS_PER_SAMPLE)
    yr  = years_by_line[ln]
    sum_pts[yr] += pts; count[yr] += 1

avg = {yr: sum_pts[yr] / count[yr] for yr in sum_pts}
if not avg:
    print("⚠️  No data – check DATE tags, ranges or file alignment."); exit()

# ── STEP 4: plot ────────────────────────────────────────────────
years = np.array(sorted(avg))
vals  = np.array([avg[y] for y in years])

coef_lin  = np.polyfit(years, vals, 1)
trend_lin = coef_lin[0] * years + coef_lin[1]

coef_poly  = np.polyfit(years, vals, POLY_DEGREE)
trend_poly = np.polyval(coef_poly, years)

plt.figure(figsize=(12, 5))
plt.bar(years, vals, width=0.8, label="Yearly average")
plt.plot(years, trend_lin,  color="red",   linewidth=2, label="Linear trend")
plt.plot(years, trend_poly, color="yellow", linewidth=3,
         label=f"Poly trend (deg {POLY_DEGREE})")
plt.title(f"Average benchmark points per year  (β={LEVEL_PENALTY}, cutoff={ERROR_CUTOFF} K)")
plt.xlabel("Year"); plt.ylabel("Average points")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=300); plt.close()
print("✅  saved", OUT_PLOT)