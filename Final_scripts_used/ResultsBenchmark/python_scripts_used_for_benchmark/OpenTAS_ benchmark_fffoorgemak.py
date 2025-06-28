#!/usr/bin/env python3
"""
Versatile benchmark for near-surface air-temperature predictions.

Changes in this version (2025-06-16)
-------------------------------------
● Added `ALT_PENALTY_LEVELS = [1, 3, 5, 7, 10, 15]` – the set of extra β
  values for which an *overall* score is now reported.

● `calculate_metrics_for_scenario()` now accepts an `alt_penalties` list
  (defaults to ALT_PENALTY_LEVELS) and computes
  `metrics["alt_penalty_scores"]`, a `{β: percentage}` mapping.

● `format_report()` prints a new section  
  ▶ SCORE WITH DIFFERENT PENALTY  
  immediately after the “SCORE EXCLUDING WORST X%” analysis.

Everything else – including CLI usage – is backwards-compatible.
"""

from pathlib import Path
import argparse
import json
import math
import re
import statistics
import sys
from typing import List, Optional, Dict, Any, Tuple
from collections import OrderedDict

# ╔═══════════════════════════════════════════════════╗
# 0 ─ DEFAULT FILE PATHS & SETTINGS
# ╚═══════════════════════════════════════════════════╝
DEFAULT_TRUTH_FILE             = Path("/Users/yer/scriptie/Final_folder_scriptieV1/Data_Used_Model_Training_V2_ChatML/Test_data_sets_semantic/1pctCO2_increase_semantic/Non_labeled_test_1pctCO2_dataset/Non_labeled_test_1pctCO2_dataset.chatml.jsonl") 
DEFAULT_PREDICTIONS_FILE       = Path("/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/1pctco2_data/infer_7B_bs4_interval1_allYears_compiled_12431794.jsonl") 
DEFAULT_OUT_REPORT_FILE        = Path("/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/1pctco2_data/1pctco2_test_7B_openTAS_NON_labeled_RESULTS.txt")
DEFAULT_DETAILED_SCORES_FILE   = Path("/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/1pctco2_data/1pctco2_data1pctco2_test_7B_openTAS_NON_labeled_DEFAULT_DETAILED_SCORES_FILE.jsonl")
DEFAULT_FILTERED_TRUTH_OUTPUT_FILE = Path("/Users/yer/scriptie/Final_folder_scriptieV1/ResultsBenchmark/7BopenTAS/non_labeled/fine_tuned_model/1pctco2_data/1pctco2_test_7B_openTAS_NON_labeled_DEFAULT_FILTERED_TRUTH_OUTPUT_FILE.jsonl") # Optional output

#--- Benchmark Parameters (can be overridden by CLI) ---
POINTS_PER_SAMPLE        = 100
LEVEL_PENALTY            = 8           # default β
ERROR_CUTOFF             = 5.0         # K
ALLOW_CLI_OVERRIDE       = True
SAVE_FILTERED_TRUTHS     = True
DEFAULT_MONTH_INTERVAL   = 1           # sample every nth month

ALT_PENALTY_LEVELS       = [1, 3, 5, 7, 10, 15, 20]  # NEW: extra β’s to report
# ════════════════════════════════════════════════════

date_regex = re.compile(r"DATE=(\d{4})-(\d{2})")
_RX_KELVIN   = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:kelvin|k)\b", re.I)
_RX_CELSIUS  = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°C|C|Celsius)\b", re.I)
_RX_PHRASE_VALUE = re.compile(
    r"(?i)(?:temperature is|air temperature is|value is|temperature will be|temperature of|temperature will reach|temperature[^0-9\-]{0,20}?is|is around|is approximately|is about|is roughly)[^0-9\-]{0,20}?(-?\d+(?:\.\d+)?)")
_RX_IS_VALUE = re.compile(r"(?i)\bis\s*:?\s*(-?\d+(?:\.\d+)?)")
_RX_NUMBER   = re.compile(r"-?\d+(?:\.\d+)?")

PLAUDIBLE_MIN_K, PLAUDIBLE_MAX_K = 150.0, 400.0
PLAUDIBLE_MIN_C, PLAUSIBLE_MAX_C = -90.0, 70.0


# ────────────────────────────────────────────────────
# 1.  BASIC HELPERS
# ────────────────────────────────────────────────────
def celsius_to_kelvin(c: float) -> float: return c + 273.15


def _parse_date_from_user_content(user_content: str) -> Optional[Tuple[int, int]]:
    m = date_regex.search(user_content)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return None


def _extract_value_from_text(text: str, parse_mode: str) -> Optional[float]:
    if not isinstance(text, str):
        return None

    # 1) explicit K or °C
    if parse_mode == "kelvin_and_celsius":
        for m in _RX_KELVIN.finditer(text):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if PLAUDIBLE_MIN_K <= v <= PLAUDIBLE_MAX_K:
                return v
        for m in _RX_CELSIUS.finditer(text):
            try:
                v_c = float(m.group(1))
            except ValueError:
                continue
            if PLAUDIBLE_MIN_C <= v_c <= PLAUSIBLE_MAX_C:
                v_k = celsius_to_kelvin(v_c)
                if PLAUDIBLE_MIN_K <= v_k <= PLAUDIBLE_MAX_K:
                    return v_k

    elif parse_mode == "kelvin_strict":
        for m in _RX_KELVIN.finditer(text):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if PLAUDIBLE_MIN_K <= v <= PLAUDIBLE_MAX_K:
                return v

    # 2) contextual “temperature is …” phrases
    for m in _RX_PHRASE_VALUE.finditer(text):
        try:
            v_num = float(m.group(1))
        except ValueError:
            continue
        context = m.group(0).lower()

        if parse_mode == "kelvin_and_celsius" and ("°c" in context or "celsius" in context):
            if PLAUDIBLE_MIN_C <= v_num <= PLAUSIBLE_MAX_C:
                v_k = celsius_to_kelvin(v_num)
                if PLAUDIBLE_MIN_K <= v_k <= PLAUDIBLE_MAX_K:
                    return v_k
            continue

        if "k" in context or "kelvin" in context:
            if PLAUDIBLE_MIN_K <= v_num <= PLAUDIBLE_MAX_K:
                return v_num
        elif PLAUDIBLE_MIN_K <= v_num <= PLAUDIBLE_MAX_K:
            return v_num

    # 3) fallback bare numbers
    if parse_mode == "kelvin_and_celsius":
        for m in _RX_IS_VALUE.finditer(text):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if PLAUDIBLE_MIN_K <= v <= PLAUDIBLE_MAX_K:
                return v

        nums = []
        for m in _RX_NUMBER.finditer(text):
            num = m.group(0)
            suffix = text[m.end(): m.end() + 5].strip().lower()
            if suffix.startswith(("ppm", "%", "hpa", "μm", "kg", "w m", "°c", "c")):
                continue
            try:
                v = float(num)
            except ValueError:
                continue
            if PLAUDIBLE_MIN_K <= v <= PLAUDIBLE_MAX_K:
                nums.append(v)
        if len(nums) == 1:
            return nums[0]

    return None


def _get_assistant_content_from_truth_obj(obj: dict) -> Optional[str]:
    if "messages" in obj and isinstance(obj["messages"], list):
        for msg in reversed(obj["messages"]):
            if msg.get("role") == "assistant":
                return msg.get("content")
    return None


def _truth_from_obj(obj: dict, ln: int, fpath: Path) -> Optional[float]:
    a_content = _get_assistant_content_from_truth_obj(obj)
    if isinstance(a_content, str):
        val = _extract_value_from_text(a_content, "kelvin_and_celsius")
        if val is not None:
            return val
        # bare number without unit
        m = re.search(r"(?:value is|temperature is)\s*(-?\d+(?:\.\d+)?)(?!\s*(?:K|Kelvin|°C|Celsius))",
                      a_content, re.I)
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                pass
            else:
                if PLAUDIBLE_MIN_C <= v <= PLAUSIBLE_MAX_C:
                    return celsius_to_kelvin(v)
                if PLAUDIBLE_MIN_K <= v <= PLAUDIBLE_MAX_K:
                    return v

    for k in ("truth_value", "target_kelvin", "value", "temperature", "label"):
        if k in obj:
            v = obj[k]
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    v2 = _extract_value_from_text(v, "kelvin_and_celsius")
                    if v2 is not None:
                        return v2

    print(f"Warning: could not extract truth (line {ln})", file=sys.stderr)
    return None


# ────────────────────────────────────────────────────
# 2.  FILE READERS – **pair by line index**
# ────────────────────────────────────────────────────
def _read_truth_file(path: Path,
                     month_interval: int,
                     stop_year: Optional[int]) -> Dict[str, Dict[str, Any]]:
    truth_map: Dict[str, Dict[str, Any]] = {}
    skipped_intvl = skipped_year = 0
    total_lines = 0

    with path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):            # 1-based index
            total_lines = ln
            try:
                obj = json.loads(line)
                # optional sampling by month / stop-year
                user_content = ""
                for msg in obj.get("messages", []):
                    if msg.get("role") == "user":
                        user_content = msg.get("content", "")
                        break
                year, month = _parse_date_from_user_content(user_content)

                if stop_year is not None and year and year > stop_year:
                    skipped_year += 1
                    continue
                if month_interval > 1:
                    if year is None or month is None:
                        print(f"Warning: no DATE on truth line {ln}; keeping it.", file=sys.stderr)
                    elif (month - 1) % month_interval != 0:
                        skipped_intvl += 1
                        continue

                pid = f"line_{ln}"                   # unique ID by line number
                truth_map[pid] = {
                    "truth_value_k": _truth_from_obj(obj, ln, path),
                    "original_truth_data": obj,
                    "truth_raw_assistant_text": _get_assistant_content_from_truth_obj(obj)
                }

            except json.JSONDecodeError:
                print(f"Error: line {ln} in {path} is not valid JSONL – skipped", file=sys.stderr)
            except Exception as e:
                print(f"Error processing truth line {ln}: {e} – skipped", file=sys.stderr)

    print(
        f"INFO: '{path.name}': read {total_lines} lines; "
        f"selected {len(truth_map)} after interval ({month_interval}) / stop-year; "
        f"skipped by interval: {skipped_intvl}; skipped by stop_year: {skipped_year}.",
        file=sys.stderr
    )
    return truth_map


def _read_predictions_file_all_modes(path: Path) -> Dict[str, Dict[str, Any]]:
    preds: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
                raw = obj.get("prediction_raw_text")
                if raw is None:
                    continue

                pid = f"line_{ln}"                   # same scheme as truth
                preds[pid] = {
                    "pred_value_k_with_c_conv": _extract_value_from_text(raw, "kelvin_and_celsius"),
                    "pred_value_k_strict": _extract_value_from_text(raw, "kelvin_strict"),
                    "prediction_raw_text": raw
                }
            except Exception as e:
                print(f"Error processing prediction line {ln}: {e} – skipped", file=sys.stderr)
    return preds


# ────────────────────────────────────────────────────
# 3.  SCORING UTILITIES
# ────────────────────────────────────────────────────
def _log_penalty(err: float, beta: float, cutoff: float, pts: int) -> float:
    if math.isnan(err) or abs(err) >= cutoff:
        return 0.0
    if beta <= 0:
        return pts * (1.0 - abs(err) / cutoff)
    up = math.log1p(beta * cutoff)
    return pts * (1.0 - math.log1p(beta * abs(err)) / up)


def _r2(y_true: List[float], y_pred: List[float]) -> float:
    if len(y_true) < 2:
        return float("nan")
    mean = statistics.mean(y_true)
    ss_tot = sum((y - mean) ** 2 for y in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else float("-inf")
    return 1.0 - ss_res / ss_tot


# ────────────────────────────────────────────────────
# 4.  METRICS CALCULATION
# ────────────────────────────────────────────────────
def calculate_metrics_for_scenario(
    truth_map: Dict[str, Dict[str, Any]],
    pred_map: Dict[str, Optional[float]],
    all_pred_info: Dict[str, Dict[str, Any]],
    beta: float,
    cutoff: float,
    pts_per_sample: int,
    detailed_writer,
    scenario_name: str,
    is_primary_log: bool,
    alt_penalties: Optional[List[int]] = None
) -> Dict[str, Any]:

    if alt_penalties is None:
        alt_penalties = ALT_PENALTY_LEVELS

    results = []
    unparsable_pred = 0

    for pid, tinfo in truth_map.items():
        truth_k = tinfo["truth_value_k"]
        pred_k = pred_map.get(pid)

        err = score = float("nan")
        if truth_k is not None and pred_k is not None:
            err = abs(truth_k - pred_k)
            score = _log_penalty(err, beta, cutoff, pts_per_sample)
        elif pred_k is None:
            unparsable_pred += 1

        results.append({
            "id": pid, "truth": truth_k, "pred": pred_k,
            "error": err, "score": score
        })

        # detailed log (once per ID, from primary scenario)
        if is_primary_log and detailed_writer:
            all_info = all_pred_info.get(pid, {})
            log = OrderedDict()
            log["id"] = pid
            log["truth_parsed_k"] = truth_k
            log["prediction_parsed_k_with_c_conv"] = all_info.get("pred_value_k_with_c_conv")
            log["prediction_parsed_k_strict"] = all_info.get("pred_value_k_strict")
            log["absolute_error_k_vs_c_conv_pred"] = (
                abs(truth_k - all_info.get("pred_value_k_with_c_conv"))
                if truth_k is not None and all_info.get("pred_value_k_with_c_conv") is not None else None
            )
            log["score_vs_c_conv_pred"] = (
                _log_penalty(log["absolute_error_k_vs_c_conv_pred"], beta, cutoff, pts_per_sample)
                if log["absolute_error_k_vs_c_conv_pred"] is not None else 0.0
            )
            log["absolute_error_k_vs_strict_pred"] = (
                abs(truth_k - all_info.get("pred_value_k_strict"))
                if truth_k is not None and all_info.get("pred_value_k_strict") is not None else None
            )
            log["score_vs_pred_strict"] = (
                _log_penalty(log["absolute_error_k_vs_strict_pred"], beta, cutoff, pts_per_sample)
                if log["absolute_error_k_vs_strict_pred"] is not None else 0.0
            )
            log["truth_raw_assistant_text"] = tinfo["truth_raw_assistant_text"]
            log["prediction_raw_text"] = all_info.get("prediction_raw_text")
            detailed_writer.write(json.dumps(log) + "\n")

    # overall stats
    scorable = [r for r in results if r["truth"] is not None]
    total_possible = len(scorable) * pts_per_sample
    total_points = sum(r["score"] for r in scorable if not math.isnan(r["score"]))
    final_pct = 100.0 * total_points / total_possible if total_possible else 0.0

    parsed_pairs = [r for r in scorable if r["pred"] is not None]
    errs = [r["error"] for r in parsed_pairs if not math.isnan(r["error"])]

    alt_scores = {}
    for b in alt_penalties:
        pts = sum(
            _log_penalty(e, b, cutoff, pts_per_sample)
            for e in errs
        )
        alt_scores[b] = 100.0 * pts / total_possible if total_possible else 0.0

    nz = [r for r in scorable if r["score"] > 0]
    score_nz = (100.0 * sum(r["score"] for r in nz) / (len(nz) * pts_per_sample)
                if nz else 0.0)

    return {
        "scorable": scorable,
        "parsed_pairs": parsed_pairs,
        "total_points": total_points,
        "total_possible": total_possible,
        "final_pct": final_pct,
        "mae": statistics.mean(errs) if errs else float("nan"),
        "rmse": math.sqrt(statistics.mean(e ** 2 for e in errs)) if errs else float("nan"),
        "r2": _r2([r["truth"] for r in parsed_pairs], [r["pred"] for r in parsed_pairs]),
        "outliers": sum(e >= cutoff for e in errs),
        "unparsable_pred": unparsable_pred,
        "score_nz": score_nz,
        "nz_count": len(nz),
        "alt_scores": alt_scores,
        "all_results": results
    }


# ────────────────────────────────────────────────────
# 5.  REPORT GENERATION
# ────────────────────────────────────────────────────
def _append_report_section(lines: List[str], title: str, m: Dict[str, Any],
                           beta: float, cutoff: float, pts_per_sample: int) -> None:
    lines.append(f"--- {title} ---\n")
    lines.append(f"Samples benchmarked: {len(m['scorable'])}\n")
    lines.append(f"  - Predictions parsed: {len(m['parsed_pairs'])}\n")
    lines.append(f"  - Unparsable predictions: {m['unparsable_pred']}\n")
    lines.append(f"Penalty β  : {beta}\nCut-off    : {cutoff} K\nPts/sample : {pts_per_sample}\n\n")

    lines.append("▶ OVERALL RESULT\n")
    lines.append(f"  Score: {m['total_points']:.1f} / {m['total_possible']:.0f} pts  →  {m['final_pct']:.2f}%\n\n")

    lines.append("▶ AGGREGATE METRICS\n")
    lines.append(f"  MAE  : {m['mae']:.4f} K\n")
    lines.append(f"RMSE : {m['rmse']:.4f} K\n")
    lines.append(f"R²   : {m['r2']:.4f}\n")
    lines.append(f"  Outliers (error ≥ {cutoff} K) : {m['outliers']} / {len(m['parsed_pairs'])}\n\n")

    lines.append("▶ SCORE EXCLUDING ZERO-POINT SAMPLES\n")
    lines.append(f"  Score: {m['score_nz']:.2f}% on {m['nz_count']} samples\n\n")

    # worst-X-percent analysis
    sorted_by_score = sorted(m["all_results"], key=lambda r: r["score"])
    lines.append("▶ SCORE EXCLUDING WORST X%\n")
    n = len(sorted_by_score)
    for pct in range(10, 91, 10):
        k = math.ceil(n * pct / 100)
        keep = sorted_by_score[k:]
        if not keep:
            lines.append(f"  Excluding worst {pct}%: no samples left\n")
            continue
        pts = sum(r["score"] for r in keep)
        pct_score = 100 * pts / (len(keep) * pts_per_sample)
        lines.append(f"  Excluding worst {pct}% ({k} samples): {pct_score:.2f}% on {len(keep)} samples\n")

    # new section: alt penalties
    lines.append("\n▶ SCORE WITH DIFFERENT PENALTY\n")
    for b in sorted(m["alt_scores"]):
        lines.append(f"  Penalty {b}: {m['alt_scores'][b]:.2f}%\n")
    lines.append("\n")


# ────────────────────────────────────────────────────
# 6.  EVALUATION PIPELINE
# ────────────────────────────────────────────────────
def evaluate_main(truth_f: Path, pred_f: Path,
                  beta: float, cutoff: float, out_f: Path,
                  pts_per_sample: int, detailed_f: Path,
                  filtered_truth_f: Optional[Path],
                  month_interval: int, stop_year: Optional[int]) -> None:

    print(f"Benchmarking with:\n  Truth='{truth_f}'\n  Preds='{pred_f}'\n  Out='{out_f}'")
    print(f"Truth sampling: month interval={month_interval}, stop year={stop_year}")
    print(f"Detailed scores log: {detailed_f}")

    if not truth_f.exists():
        print(f"ERROR: truth file not found: {truth_f}")
        return
    if not pred_f.exists():
        print(f"ERROR: predictions file not found: {pred_f}")
        return

    truth_map = _read_truth_file(truth_f, month_interval, stop_year)
    pred_info = _read_predictions_file_all_modes(pred_f)

    # shrink prediction maps to ids present in truth set
    preds_c_conv = {pid: pred_info[pid]["pred_value_k_with_c_conv"]
                    for pid in truth_map if pid in pred_info}
    preds_strict = {pid: pred_info[pid]["pred_value_k_strict"]
                    for pid in truth_map if pid in pred_info}

    report_lines: List[str] = ["Versatile Temperature-Prediction Benchmark Report\n",
                               "===============================================\n\n"]

    with detailed_f.open("w", encoding="utf-8") as dsw:
        # header comment
        dsw.write("# Detailed per-sample scores (JSONL)\n")

        # scenario 1 – C→K conversion
        print("\n--- Calculating metrics (C→K conversion) ---")
        m1 = calculate_metrics_for_scenario(
            truth_map, preds_c_conv, pred_info,
            beta, cutoff, pts_per_sample,
            detailed_writer=dsw,
            scenario_name="C_to_K",
            is_primary_log=True,
            alt_penalties=ALT_PENALTY_LEVELS
        )
        _append_report_section(report_lines, "RESULTS (Celsius→Kelvin Conversion)", m1,
                               beta, cutoff, pts_per_sample)

        # scenario 2 – Kelvin-strict
        print("\n--- Calculating metrics (Kelvin-strict) ---")
        m2 = calculate_metrics_for_scenario(
            truth_map, preds_strict, pred_info,
            beta, cutoff, pts_per_sample,
            detailed_writer=None,
            scenario_name="Kelvin_Strict",
            is_primary_log=False,
            alt_penalties=ALT_PENALTY_LEVELS
        )
        _append_report_section(report_lines, "RESULTS (Kelvin-Strict Parsing)", m2,
                               beta, cutoff, pts_per_sample)

    # filtered truths
    if SAVE_FILTERED_TRUTHS and filtered_truth_f:
        with filtered_truth_f.open("w", encoding="utf-8") as ft:
            for v in truth_map.values():
                ft.write(json.dumps(v["original_truth_data"]) + "\n")
        print(f"\nSaved {len(truth_map)} sampled truths to {filtered_truth_f}")

    out_f.write_text("".join(report_lines), encoding="utf-8")
    print("\n✅  Report written to", out_f)
    print("ℹ️  Detailed scores written to", detailed_f)
    print(f"Results (C→K): {m1['final_pct']:.2f}%   |   Results (Kelvin-strict): {m2['final_pct']:.2f}%")


# ────────────────────────────────────────────────────
# 7.  CLI ENTRY POINT
# ────────────────────────────────────────────────────
def main_cli() -> None:
    ap = argparse.ArgumentParser(description="Versatile temperature benchmark")
    ap.add_argument("--truth-file", type=Path, default=DEFAULT_TRUTH_FILE)
    ap.add_argument("--pred-file",  type=Path, default=DEFAULT_PREDICTIONS_FILE)
    ap.add_argument("--out-file",   type=Path, default=DEFAULT_OUT_REPORT_FILE)
    ap.add_argument("--detailed-scores-file", type=Path, default=DEFAULT_DETAILED_SCORES_FILE)
    ap.add_argument("--filtered-truth-out-file", type=Path, default=DEFAULT_FILTERED_TRUTH_OUTPUT_FILE)
    ap.add_argument("--month-interval-truth", type=int, default=DEFAULT_MONTH_INTERVAL,
                    help="Sample every Nth month from the truth file")
    ap.add_argument("--stop-year-truth", type=int, default=None,
                    help="Ignore truth entries with DATE year > this")
    ap.add_argument("--level", type=int, default=LEVEL_PENALTY, help="β penalty parameter")
    ap.add_argument("--cutoff", type=float, default=ERROR_CUTOFF, help="Maximum error that still scores > 0")
    ap.add_argument("--points", type=int, default=POINTS_PER_SAMPLE, help="Points for perfect prediction")

    args = ap.parse_args([] if not ALLOW_CLI_OVERRIDE else None)

    evaluate_main(
        args.truth_file, args.pred_file,
        float(args.level), args.cutoff, args.out_file,
        args.points, args.detailed_scores_file,
        args.filtered_truth_out_file if SAVE_FILTERED_TRUTHS else None,
        args.month_interval_truth, args.stop_year_truth
    )


if __name__ == "__main__":
    main_cli()