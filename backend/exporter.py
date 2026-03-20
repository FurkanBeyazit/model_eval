"""
exporter.py  –  Excel Export  (xlsxwriter engine)
Multi-sheet workbook from validation & prediction results.

Sheets:
  1.  Summary              – overall metrics + GT totals if has_gt
  2.  Val_Class_Metrics    – P / R / F1 / mAP per class (from model.val)
  3.  Class_Performance    – TP / FP / FN / Recall / Precision per class (GT-based)
  4.  Per_Image            – detection counts + GT counts per image
  5.  Per_Image_GT         – per image × class: GT / Pred / Missed / Extra (GT only)
  6.  All_Detections       – every bbox with class, conf, coords, size
  7.  Class_Distribution   – aggregated confidence stats per class
  8.  Confusion_Matrix     – actual vs predicted class (from model.val)
  9.  Threshold_Curve      – P / R / F1 at conf 0.10→0.95  +  F1 line chart
  10. Size_Analysis        – small / medium / large with GT Recall & Avg IoU
  11. Worst_Images         – images sorted by FN (or zero-detection if no GT)
  12. Spatial_Bias         – 3×3 grid with GT overlay per class
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
from datetime import datetime

# Object size thresholds (pixels²)
_SMALL  = 32 * 32    # <  1 024
_MEDIUM = 96 * 96    # <  9 216
# large  >= 9 216

# ── Colour hex codes for xlsxwriter ──────────────────────────────────────────
_C_GREEN  = "#C6EFCE"
_C_RED    = "#FFC7CE"
_C_ORANGE = "#FFEB9C"
_C_BLUE   = "#BDD7EE"
_C_GREY   = "#D9D9D9"
_C_DKBLUE = "#2F75B6"
_C_WHITE  = "#FFFFFF"


def export_to_excel(
    predict_results: List[dict],
    val_metrics: Optional[dict],
    class_names: Dict[int, str],
    threshold_results: Optional[List[dict]] = None,
    has_gt: bool = False,
    output_path: Optional[str] = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = str(Path(tempfile.gettempdir()) / f"model_eval_{timestamp}.xlsx")

    all_classes  = sorted(set(class_names.values()))
    total_images = len(predict_results)
    total_dets   = sum(r["total_detections"] for r in predict_results)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        wb = writer.book

        # ── Shared formats ────────────────────────────────────────────────────
        hdr_fmt  = wb.add_format({"bold": True, "bg_color": _C_BLUE,   "border": 1, "align": "center"})
        bold_fmt = wb.add_format({"bold": True})
        pct_fmt  = wb.add_format({"num_format": "0.00%"})
        dec_fmt  = wb.add_format({"num_format": "0.0000"})
        int_fmt  = wb.add_format({"num_format": "0"})
        red_fmt  = wb.add_format({"bg_color": _C_RED})
        grn_fmt  = wb.add_format({"bg_color": _C_GREEN})
        org_fmt  = wb.add_format({"bg_color": _C_ORANGE})
        grey_fmt = wb.add_format({"bg_color": _C_GREY, "align": "center"})
        ctr_fmt  = wb.add_format({"align": "center"})

        # ── 1. Summary ────────────────────────────────────────────────────────
        summary_rows = [
            {"Metric": "Export Timestamp",  "Value": timestamp},
            {"Metric": "Total Images",       "Value": total_images},
            {"Metric": "Total Detections",   "Value": total_dets},
        ]
        if has_gt and predict_results:
            total_gt = sum(sum(r.get("gt_counts", {}).values()) for r in predict_results)
            total_tp = sum(r.get("tp", 0) for r in predict_results)
            total_fp = sum(r.get("fp", 0) for r in predict_results)
            total_fn = sum(r.get("fn", 0) for r in predict_results)
            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            summary_rows += [
                {"Metric": "Total GT Objects",   "Value": total_gt},
                {"Metric": "True Positives (TP)", "Value": total_tp},
                {"Metric": "False Positives (FP)", "Value": total_fp},
                {"Metric": "False Negatives (FN)", "Value": total_fn},
                {"Metric": "Overall Precision",    "Value": round(prec, 4)},
                {"Metric": "Overall Recall",       "Value": round(rec, 4)},
                {"Metric": "Overall F1",           "Value": round(f1, 4)},
            ]
        if val_metrics:
            summary_rows += [
                {"Metric": "mAP50",          "Value": val_metrics.get("map50",    "-")},
                {"Metric": "mAP50-95",       "Value": val_metrics.get("map50_95", "-")},
                {"Metric": "Mean Precision",  "Value": val_metrics.get("mp",      "-")},
                {"Metric": "Mean Recall",     "Value": val_metrics.get("mr",      "-")},
            ]
        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_excel(writer, sheet_name="Summary", index=False)
        _apply_header_fmt(writer.sheets["Summary"], wb, ["Metric", "Value"])

        # ── 2. Val_Class_Metrics ──────────────────────────────────────────────
        if val_metrics and val_metrics.get("class_metrics"):
            df_cls = pd.DataFrame(val_metrics["class_metrics"])
            df_cls.to_excel(writer, sheet_name="Val_Class_Metrics", index=False)
            ws = writer.sheets["Val_Class_Metrics"]
            _apply_header_fmt(ws, wb, df_cls.columns.tolist())
            ws.autofilter(0, 0, len(df_cls), len(df_cls.columns) - 1)
            # Color scale on mAP50
            if "mAP50" in df_cls.columns:
                col_idx = df_cls.columns.tolist().index("mAP50")
                col_letter = _col_letter(col_idx)
                ws.conditional_format(
                    f"{col_letter}2:{col_letter}{len(df_cls)+1}",
                    {"type": "3_color_scale",
                     "min_color": _C_RED, "mid_color": _C_ORANGE, "max_color": _C_GREEN,
                     "min_type": "num", "min_value": 0,
                     "mid_type": "num", "mid_value": 0.4,
                     "max_type": "num", "max_value": 1},
                )

        # ── 3. Class_Performance (GT-based) ───────────────────────────────────
        if has_gt and predict_results:
            _write_class_performance(writer, wb, predict_results, all_classes)

        # ── 4. Per_Image ──────────────────────────────────────────────────────
        if predict_results:
            rows = []
            for r in predict_results:
                row: dict = {
                    "Image":      r["image_name"],
                    "Image_Path": r["image_path"],
                    "Total_Pred": r["total_detections"],
                }
                if has_gt:
                    row["Total_GT"] = sum(r.get("gt_counts", {}).values())
                    row["TP"] = r.get("tp", "")
                    row["FP"] = r.get("fp", "")
                    row["FN"] = r.get("fn", "")
                    row["Match_Rate"] = r.get("match_rate", "")
                for cls in all_classes:
                    row[f"{cls}_pred"] = r["class_counts"].get(cls, 0)
                    if has_gt:
                        row[f"{cls}_gt"] = r.get("gt_counts", {}).get(cls, 0)
                    avg = r["class_avg_conf"].get(cls)
                    row[f"{cls}_avg_conf"] = avg if avg is not None else ""
                rows.append(row)
            df_pi = pd.DataFrame(rows)
            df_pi.to_excel(writer, sheet_name="Per_Image", index=False)
            ws = writer.sheets["Per_Image"]
            _apply_header_fmt(ws, wb, df_pi.columns.tolist())
            ws.autofilter(0, 0, len(df_pi), len(df_pi.columns) - 1)
            # Highlight FN > 0 in orange
            if has_gt and "FN" in df_pi.columns:
                fn_col = df_pi.columns.tolist().index("FN")
                fn_letter = _col_letter(fn_col)
                ws.conditional_format(
                    f"{fn_letter}2:{fn_letter}{len(df_pi)+1}",
                    {"type": "cell", "criteria": ">", "value": 0, "format": org_fmt},
                )

        # ── 5. Per_Image_GT ───────────────────────────────────────────────────
        if has_gt and predict_results:
            _write_per_image_gt(writer, wb, predict_results, all_classes)

        # ── 6. All_Detections ─────────────────────────────────────────────────
        if predict_results:
            all_dets = []
            for r in predict_results:
                for det in r["detections"]:
                    all_dets.append({
                        "image":      r["image_name"],
                        "class":      det["class_name"],
                        "confidence": det["confidence"],
                        "x1": det["x1"], "y1": det["y1"],
                        "x2": det["x2"], "y2": det["y2"],
                        "width":  det["width"],
                        "height": det["height"],
                        "area":   det["area"],
                        "cx_norm": det.get("cx_norm", ""),
                        "cy_norm": det.get("cy_norm", ""),
                    })
            if all_dets:
                df_dets = pd.DataFrame(all_dets)
                df_dets.to_excel(writer, sheet_name="All_Detections", index=False)
                ws = writer.sheets["All_Detections"]
                _apply_header_fmt(ws, wb, df_dets.columns.tolist())
                ws.autofilter(0, 0, len(df_dets), len(df_dets.columns) - 1)

        # ── 7. Class_Distribution ─────────────────────────────────────────────
        if predict_results:
            dist: Dict[str, List[float]] = {cls: [] for cls in all_classes}
            for r in predict_results:
                for det in r["detections"]:
                    cn = det["class_name"]
                    if cn in dist:
                        dist[cn].append(det["confidence"])
            dist_rows = []
            for cls in all_classes:
                confs = dist[cls]
                dist_rows.append({
                    "Class":            cls,
                    "Total_Detections": len(confs),
                    "Images_With_Det":  sum(1 for r in predict_results if cls in r["class_counts"]),
                    "Avg_Confidence":   round(sum(confs) / len(confs), 4) if confs else 0,
                    "Min_Confidence":   round(min(confs), 4) if confs else 0,
                    "Max_Confidence":   round(max(confs), 4) if confs else 0,
                })
            df_dist = pd.DataFrame(dist_rows)
            df_dist.to_excel(writer, sheet_name="Class_Distribution", index=False)
            ws = writer.sheets["Class_Distribution"]
            _apply_header_fmt(ws, wb, df_dist.columns.tolist())

        # ── 8. Confusion_Matrix ───────────────────────────────────────────────
        if val_metrics and val_metrics.get("confusion_matrix"):
            _write_confusion_matrix(writer, wb, val_metrics["confusion_matrix"])

        # ── 9. Threshold_Curve + F1 chart ─────────────────────────────────────
        if threshold_results:
            _write_threshold_curve(writer, wb, threshold_results, has_gt)

        # ── 10. Size_Analysis ─────────────────────────────────────────────────
        if predict_results:
            _write_size_analysis(writer, wb, predict_results, all_classes, has_gt)

        # ── 11. Worst_Images ──────────────────────────────────────────────────
        if predict_results:
            _write_worst_images(writer, wb, predict_results, all_classes, has_gt, red_fmt, grn_fmt, org_fmt)

        # ── 12. Spatial_Bias ──────────────────────────────────────────────────
        if predict_results:
            _write_spatial_bias(writer, wb, predict_results, all_classes, has_gt)

    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col_letter(zero_based_idx: int) -> str:
    """Convert 0-based column index to Excel letter (A, B, ... Z, AA, ...)."""
    result = ""
    n = zero_based_idx + 1
    while n:
        n, r = divmod(n - 1, 26)
        result = chr(65 + r) + result
    return result


def _apply_header_fmt(ws, wb, columns: list):
    hdr_fmt = wb.add_format({"bold": True, "bg_color": _C_BLUE, "border": 1, "align": "center"})
    for i, col in enumerate(columns):
        ws.write(0, i, col, hdr_fmt)
    # Auto-width approximation
    for i, col in enumerate(columns):
        ws.set_column(i, i, max(12, len(str(col)) + 4))


# ── Sheet writers ─────────────────────────────────────────────────────────────

def _write_class_performance(writer, wb, predict_results: List[dict], all_classes: List[str]):
    """Per-class TP / FP / FN / Recall / Precision / F1 computed from GT matching."""
    cls_tp: Dict[str, int] = {c: 0 for c in all_classes}
    cls_fp: Dict[str, int] = {c: 0 for c in all_classes}
    cls_fn: Dict[str, int] = {c: 0 for c in all_classes}
    cls_gt: Dict[str, int] = {c: 0 for c in all_classes}

    for r in predict_results:
        for pair in r.get("tp_pairs", []):
            cn = pair["pred"]["class_name"]
            if cn in cls_tp:
                cls_tp[cn] += 1
        for p in r.get("fp_preds", []):
            cn = p["class_name"]
            if cn in cls_fp:
                cls_fp[cn] += 1
        for g in r.get("fn_gts", []):
            cn = g["class_name"]
            if cn in cls_fn:
                cls_fn[cn] += 1
        for cn, cnt in r.get("gt_counts", {}).items():
            if cn in cls_gt:
                cls_gt[cn] += cnt

    rows = []
    for cls in all_classes:
        tp = cls_tp[cls]; fp = cls_fp[cls]; fn = cls_fn[cls]; gt = cls_gt[cls]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({
            "Class":     cls,
            "GT_Count":  gt,
            "TP":        tp,
            "FP":        fp,
            "FN":        fn,
            "Precision": round(prec, 4),
            "Recall":    round(rec, 4),
            "F1":        round(f1, 4),
        })

    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name="Class_Performance", index=False)
    ws = writer.sheets["Class_Performance"]
    _apply_header_fmt(ws, wb, df.columns.tolist())
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    # Color scale on Recall and F1
    for col_name in ("Recall", "F1", "Precision"):
        if col_name in df.columns:
            ci = df.columns.tolist().index(col_name)
            cl = _col_letter(ci)
            ws.conditional_format(
                f"{cl}2:{cl}{len(df)+1}",
                {"type": "3_color_scale",
                 "min_color": _C_RED, "mid_color": _C_ORANGE, "max_color": _C_GREEN,
                 "min_type": "num", "min_value": 0,
                 "mid_type": "num", "mid_value": 0.5,
                 "max_type": "num", "max_value": 1},
            )


def _write_per_image_gt(writer, wb, predict_results: List[dict], all_classes: List[str]):
    """Per-image per-class GT vs Pred breakdown."""
    rows = []
    for r in predict_results:
        for cls in all_classes:
            gt_n   = r.get("gt_counts", {}).get(cls, 0)
            pred_n = r["class_counts"].get(cls, 0)
            # Per-class TP/FP/FN from stored pairs
            tp = sum(1 for p in r.get("tp_pairs", []) if p["pred"]["class_name"] == cls)
            fp = sum(1 for p in r.get("fp_preds", []) if p["class_name"] == cls)
            fn = sum(1 for g in r.get("fn_gts",   []) if g["class_name"]  == cls)
            if gt_n == 0 and pred_n == 0:
                continue
            match_rate = round(tp / gt_n, 4) if gt_n > 0 else None
            rows.append({
                "Image":      r["image_name"],
                "Class":      cls,
                "GT_Count":   gt_n,
                "Pred_Count": pred_n,
                "TP":         tp,
                "FP_Extra":   fp,
                "FN_Missed":  fn,
                "Match_Rate": match_rate if match_rate is not None else "",
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name="Per_Image_GT", index=False)
    ws = writer.sheets["Per_Image_GT"]
    _apply_header_fmt(ws, wb, df.columns.tolist())
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    org_fmt = wb.add_format({"bg_color": _C_ORANGE})
    red_fmt = wb.add_format({"bg_color": _C_RED})

    # Orange where FN_Missed > 0
    fn_ci = df.columns.tolist().index("FN_Missed")
    fn_cl = _col_letter(fn_ci)
    ws.conditional_format(
        f"{fn_cl}2:{fn_cl}{len(df)+1}",
        {"type": "cell", "criteria": ">", "value": 0, "format": org_fmt},
    )
    # Red where FP_Extra > 0
    fp_ci = df.columns.tolist().index("FP_Extra")
    fp_cl = _col_letter(fp_ci)
    ws.conditional_format(
        f"{fp_cl}2:{fp_cl}{len(df)+1}",
        {"type": "cell", "criteria": ">", "value": 0, "format": red_fmt},
    )


def _write_confusion_matrix(writer, wb, cm_data: dict):
    labels = cm_data["labels"]
    matrix = cm_data["matrix"]
    nc = len(labels) - 1   # background is last

    row_labels = [f"Pred: {l}" for l in labels]
    col_labels = [f"GT: {l}"   for l in labels]
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.to_excel(writer, sheet_name="Confusion_Matrix")

    ws = writer.sheets["Confusion_Matrix"]
    hdr_fmt  = wb.add_format({"bold": True, "bg_color": _C_BLUE,   "border": 1, "align": "center"})
    grn_fmt  = wb.add_format({"bg_color": _C_GREEN,  "align": "center"})
    red_fmt  = wb.add_format({"bg_color": _C_RED,    "align": "center"})
    org_fmt  = wb.add_format({"bg_color": _C_ORANGE, "align": "center"})
    grey_fmt = wb.add_format({"bg_color": _C_GREY,   "align": "center"})
    norm_fmt = wb.add_format({"align": "center"})

    n_rows = len(labels)
    n_cols = len(labels)

    for c_idx in range(n_cols + 1):  # +1 for index column
        ws.set_column(c_idx, c_idx, 14)

    # Re-write cell by cell with colour
    for r_idx in range(n_rows):
        r_label = row_labels[r_idx]
        ws.write(r_idx + 1, 0, r_label, hdr_fmt)
        for c_idx in range(n_cols):
            val = matrix[r_idx][c_idx]
            if r_idx == nc or c_idx == nc:
                fmt = org_fmt if val > 0 else grey_fmt
            elif r_idx == c_idx:
                fmt = grn_fmt
            elif val > 0:
                fmt = red_fmt
            else:
                fmt = norm_fmt
            ws.write(r_idx + 1, c_idx + 1, val, fmt)

    # Header row
    ws.write(0, 0, "", hdr_fmt)
    for c_idx, cl in enumerate(col_labels):
        ws.write(0, c_idx + 1, cl, hdr_fmt)

    # ── Bar chart: TP / FN (missed as BG) / FP per class ─────────────────────
    nc_classes = labels[:-1]   # exclude background row/col
    n_cls      = len(nc_classes)
    if n_cls > 0:
        cdata_row = n_rows + 3   # 0-indexed row for chart data header
        chart_hdr = wb.add_format({"bold": True, "bg_color": _C_GREY, "border": 1})
        ws.write(cdata_row, 0, "Class",         chart_hdr)
        ws.write(cdata_row, 1, "TP",            chart_hdr)
        ws.write(cdata_row, 2, "FN (missed→BG)", chart_hdr)
        ws.write(cdata_row, 3, "FP (BG→cls)",   chart_hdr)

        for i, cls in enumerate(nc_classes):
            tp_val = int(matrix[i][i])       if i < len(matrix) and i < len(matrix[i])  else 0
            fn_val = int(matrix[nc][i])      if nc < len(matrix) and i < len(matrix[nc]) else 0
            fp_val = int(matrix[i][nc])      if i < len(matrix) and nc < len(matrix[i]) else 0
            ws.write(cdata_row + 1 + i, 0, cls)
            ws.write(cdata_row + 1 + i, 1, tp_val)
            ws.write(cdata_row + 1 + i, 2, fn_val)
            ws.write(cdata_row + 1 + i, 3, fp_val)

        chart = wb.add_chart({"type": "bar", "subtype": "clustered"})
        for col_idx, (sname, scolor) in enumerate(
            [("TP", "#00B050"), ("FN (missed→BG)", "#FF8C00"), ("FP (BG→cls)", "#FF0000")], start=1
        ):
            chart.add_series({
                "name":       sname,
                "categories": ["Confusion_Matrix", cdata_row + 1, 0, cdata_row + n_cls, 0],
                "values":     ["Confusion_Matrix", cdata_row + 1, col_idx, cdata_row + n_cls, col_idx],
                "fill":       {"color": scolor},
            })
        chart.set_title({"name": "TP / FN (missed as BG) / FP per Class"})
        chart.set_x_axis({"name": "Count"})
        chart.set_y_axis({"name": "Class", "reverse": True})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 420, "height": max(200, 80 + n_cls * 38)})
        ws.insert_chart(cdata_row, 5, chart)


def _write_threshold_curve(writer, wb, threshold_results: List[dict], has_gt: bool):
    df = pd.DataFrame(threshold_results)
    df.to_excel(writer, sheet_name="Threshold_Curve", index=False)
    ws = writer.sheets["Threshold_Curve"]
    cols = df.columns.tolist()
    _apply_header_fmt(ws, wb, cols)
    n_rows = len(df)

    # Color scale on numeric columns (skip Threshold)
    for i, col in enumerate(cols):
        if col == "Threshold":
            continue
        cl = _col_letter(i)
        ws.conditional_format(
            f"{cl}2:{cl}{n_rows+1}",
            {"type": "2_color_scale", "min_color": _C_WHITE, "max_color": _C_DKBLUE},
        )

    # F1 / Precision / Recall line chart (only when GT data present)
    if has_gt and "F1" in cols and "Precision" in cols and "Recall" in cols:
        chart = wb.add_chart({"type": "line"})
        thresh_col = cols.index("Threshold")
        for series_name, color in [("F1", "#0070C0"), ("Precision", "#00B050"), ("Recall", "#FF0000")]:
            if series_name in cols:
                val_col = cols.index(series_name)
                chart.add_series({
                    "name":       series_name,
                    "categories": ["Threshold_Curve", 1, thresh_col, n_rows, thresh_col],
                    "values":     ["Threshold_Curve", 1, val_col,    n_rows, val_col],
                    "line":       {"color": color, "width": 2},
                    "marker":     {"type": "circle", "size": 4},
                })
        chart.set_title({"name": "F1 / Precision / Recall vs Confidence Threshold"})
        chart.set_x_axis({"name": "Confidence Threshold", "num_format": "0.00"})
        chart.set_y_axis({"name": "Score", "min": 0, "max": 1, "num_format": "0.00"})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 480, "height": 288})

        # Place chart below the data table
        chart_row = n_rows + 3
        ws.insert_chart(chart_row, 0, chart)

        # Optimal threshold annotation (max F1)
        if threshold_results:
            best = max(threshold_results, key=lambda x: x.get("F1", 0))
            note_fmt = wb.add_format({"bold": True, "bg_color": _C_GREEN, "border": 1})
            ws.write(chart_row - 1, 0, f"Optimal threshold: {best['Threshold']}  (F1={best.get('F1', 0):.4f})", note_fmt)
            ws.set_column(0, 0, 50)


def _write_size_analysis(writer, wb, predict_results: List[dict], all_classes: List[str], has_gt: bool):
    rows = []
    for cls in all_classes:
        buckets = {
            "Small":  {"dets": [], "tp_ious": []},
            "Medium": {"dets": [], "tp_ious": []},
            "Large":  {"dets": [], "tp_ious": []},
        }
        gt_counts = {"Small": 0, "Medium": 0, "Large": 0}

        for r in predict_results:
            for det in r["detections"]:
                if det["class_name"] != cls:
                    continue
                area = det["area"]
                bucket = "Small" if area < _SMALL else ("Medium" if area < _MEDIUM else "Large")
                buckets[bucket]["dets"].append(det)

            if has_gt:
                # GT sizes from fn_gts + tp_pairs gt boxes
                for pair in r.get("tp_pairs", []):
                    if pair["gt"]["class_name"] == cls:
                        area = pair["gt"]["area"]
                        bucket = "Small" if area < _SMALL else ("Medium" if area < _MEDIUM else "Large")
                        gt_counts[bucket] += 1
                        buckets[bucket]["tp_ious"].append(pair["iou"])
                for g in r.get("fn_gts", []):
                    if g["class_name"] == cls:
                        area = g["area"]
                        bucket = "Small" if area < _SMALL else ("Medium" if area < _MEDIUM else "Large")
                        gt_counts[bucket] += 1

        for bucket in ("Small", "Medium", "Large"):
            dets     = buckets[bucket]["dets"]
            tp_ious  = buckets[bucket]["tp_ious"]
            gt_n     = gt_counts[bucket]
            det_n    = len(dets)
            avg_conf = round(sum(d["confidence"] for d in dets) / det_n, 3) if dets else 0
            row = {
                "Class":          cls,
                "Size_Bucket":    bucket,
                "Det_Count":      det_n,
                "Avg_Confidence": avg_conf,
            }
            if has_gt:
                recall   = round(len(tp_ious) / gt_n, 4) if gt_n > 0 else ""
                avg_iou  = round(sum(tp_ious) / len(tp_ious), 4) if tp_ious else ""
                row["GT_Count"]  = gt_n
                row["Found_TP"]  = len(tp_ious)
                row["Recall"]    = recall
                row["Avg_IoU"]   = avg_iou
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name="Size_Analysis", index=False)
    ws = writer.sheets["Size_Analysis"]
    _apply_header_fmt(ws, wb, df.columns.tolist())
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    # Color scale on Recall if present
    if has_gt and "Recall" in df.columns:
        ci = df.columns.tolist().index("Recall")
        cl = _col_letter(ci)
        ws.conditional_format(
            f"{cl}2:{cl}{len(df)+1}",
            {"type": "3_color_scale",
             "min_color": _C_RED, "mid_color": _C_ORANGE, "max_color": _C_GREEN,
             "min_type": "num", "min_value": 0,
             "mid_type": "num", "mid_value": 0.5,
             "max_type": "num", "max_value": 1},
        )

    # ── Pivot table + clustered column chart ─────────────────────────────────
    # Build pivot: rows = classes, cols = Small / Medium / Large
    val_field  = "Recall"    if (has_gt and "Recall" in df.columns) else "Det_Count"
    y_label    = "Recall (0–1)" if val_field == "Recall" else "Detection Count"
    chart_ymax = 1.0 if val_field == "Recall" else None

    pivot_start = len(df) + 3   # 0-indexed Excel row
    phdr_fmt = wb.add_format({"bold": True, "bg_color": _C_GREY, "border": 1, "align": "center"})
    for ci, col_title in enumerate(["Class", "Small", "Medium", "Large"]):
        ws.write(pivot_start, ci, col_title, phdr_fmt)

    pivot_rows = []
    for cls in all_classes:
        cls_rows = [r for r in rows if r["Class"] == cls]
        vals = {}
        for cr in cls_rows:
            raw = cr.get(val_field, 0)
            vals[cr["Size_Bucket"]] = raw if raw != "" else 0
        pivot_rows.append({
            "cls": cls,
            "Small":  vals.get("Small",  0),
            "Medium": vals.get("Medium", 0),
            "Large":  vals.get("Large",  0),
        })
        ws.write(pivot_start + 1 + len(pivot_rows) - 1, 0, cls)
        ws.write(pivot_start + 1 + len(pivot_rows) - 1, 1, vals.get("Small",  0))
        ws.write(pivot_start + 1 + len(pivot_rows) - 1, 2, vals.get("Medium", 0))
        ws.write(pivot_start + 1 + len(pivot_rows) - 1, 3, vals.get("Large",  0))

    n_pivot = len(pivot_rows)
    if n_pivot > 0:
        chart = wb.add_chart({"type": "column", "subtype": "clustered"})
        for bucket_col, (bucket, bcolor) in enumerate(
            [("Small", "#4472C4"), ("Medium", "#ED7D31"), ("Large", "#70AD47")], start=1
        ):
            chart.add_series({
                "name":       bucket,
                "categories": ["Size_Analysis", pivot_start + 1, 0, pivot_start + n_pivot, 0],
                "values":     ["Size_Analysis", pivot_start + 1, bucket_col, pivot_start + n_pivot, bucket_col],
                "fill":       {"color": bcolor},
            })
        y_opts = {"name": y_label}
        if chart_ymax is not None:
            y_opts["max"] = chart_ymax
        chart.set_title({"name": f"Size Bucket — {val_field} per Class"})
        chart.set_x_axis({"name": "Class"})
        chart.set_y_axis(y_opts)
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 420, "height": 260})
        ws.insert_chart(pivot_start, 5, chart)


def _write_worst_images(writer, wb, predict_results: List[dict], all_classes: List[str],
                        has_gt: bool, red_fmt, grn_fmt, org_fmt):
    rows = []
    for r in predict_results:
        confs = [det["confidence"] for det in r["detections"]]
        row: dict = {
            "Image":            r["image_name"],
            "Total_Pred":       r["total_detections"],
            "Avg_Conf":         round(sum(confs) / len(confs), 3) if confs else 0,
        }
        if has_gt:
            fn = r.get("fn", 0)
            fp = r.get("fp", 0)
            tp = r.get("tp", 0)
            gt_total = sum(r.get("gt_counts", {}).values())
            row["Total_GT"] = gt_total
            row["TP"]       = tp
            row["FP"]       = fp
            row["FN_Missed"] = fn
            row["Match_Rate"] = r.get("match_rate", "")
        else:
            row["Zero_Detection"] = r["total_detections"] == 0
        for cls in all_classes:
            row[f"{cls}_pred"] = r["class_counts"].get(cls, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    if has_gt:
        df = df.sort_values(["FN_Missed", "Avg_Conf"], ascending=[False, True])
    else:
        df = df.sort_values(["Zero_Detection", "Avg_Conf"], ascending=[False, True])

    df.to_excel(writer, sheet_name="Worst_Images", index=False)
    ws = writer.sheets["Worst_Images"]
    _apply_header_fmt(ws, wb, df.columns.tolist())
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    # Highlight worst rows
    if has_gt and "FN_Missed" in df.columns:
        fn_ci = df.columns.tolist().index("FN_Missed")
        fn_cl = _col_letter(fn_ci)
        ws.conditional_format(
            f"{fn_cl}2:{fn_cl}{len(df)+1}",
            {"type": "cell", "criteria": ">", "value": 0, "format": org_fmt},
        )
    elif not has_gt and "Zero_Detection" in df.columns:
        zd_ci = df.columns.tolist().index("Zero_Detection")
        zd_cl = _col_letter(zd_ci)
        ws.conditional_format(
            f"{zd_cl}2:{zd_cl}{len(df)+1}",
            {"type": "cell", "criteria": "==", "value": 1, "format": red_fmt},
        )


def _write_spatial_bias(writer, wb, predict_results: List[dict], all_classes: List[str], has_gt: bool):
    ws = wb.add_worksheet("Spatial_Bias")
    writer.sheets["Spatial_Bias"] = ws

    hdr_fmt  = wb.add_format({"bold": True, "bg_color": _C_BLUE, "border": 1, "align": "center"})
    bold_fmt = wb.add_format({"bold": True})
    ctr_fmt  = wb.add_format({"align": "center"})
    note_fmt = wb.add_format({"italic": True, "font_color": "#595959"})

    # ── Ratio / percentage cell formats ──────────────────────────────────────
    # Ratio is stored as fraction (1.0 = 100%).  "0%" num_format multiplies ×100.
    pct_fmt  = wb.add_format({"align": "center", "num_format": "0%"})
    pct_red  = wb.add_format({"align": "center", "num_format": "0%",
                               "bg_color": _C_RED,    "bold": True})   # >130% over-detecting
    pct_org  = wb.add_format({"align": "center", "num_format": "0%",
                               "bg_color": _C_ORANGE, "bold": True})   # <70%  under-detecting
    pct_grn  = wb.add_format({"align": "center", "num_format": "0%",
                               "bg_color": _C_GREEN})                   # 70–130% ok
    pct_grey = wb.add_format({"align": "center", "bg_color": _C_GREY}) # "—"
    pct_fp   = wb.add_format({"align": "center", "bg_color": _C_RED,
                               "bold": True, "font_color": "#CC0000"})  # FP! no GT

    col_labels = ["Left", "Centre", "Right"]
    row_labels = ["Top", "Middle", "Bottom"]

    # ── Grid builders ─────────────────────────────────────────────────────────

    def _build_grid(dets):
        grid = [[0]*3 for _ in range(3)]
        for det in dets:
            c  = min(int(det.get("cx_norm", 0.5) * 3), 2)
            ro = min(int(det.get("cy_norm", 0.5) * 3), 2)
            grid[ro][c] += 1
        return grid

    def _build_gt_grid(gt_boxes):
        grid = [[0]*3 for _ in range(3)]
        for g in gt_boxes:
            c  = min(int(g.get("cx_norm", 0.5) * 3), 2)
            ro = min(int(g.get("cy_norm", 0.5) * 3), 2)
            grid[ro][c] += 1
        return grid

    def _ratio_cell(pred_n: int, gt_n: int):
        """
        Returns (value, fmt) for the Coverage Ratio cell.
        ratio = pred_n / gt_n  (fraction; 1.0 = 100%)
          > 1.30 → red   (over-detecting in this region)
          < 0.70 → orange (under-detecting / missing GT)
          0.70–1.30 → green (well-calibrated)
        Special cases:
          gt=0 & pred>0 → "FP!" red (predictions with no GT anchor)
          both 0 → "—" grey
        """
        if gt_n == 0 and pred_n == 0:
            return "—", pct_grey
        if gt_n == 0:
            return "FP!", pct_fp
        ratio = pred_n / gt_n
        if ratio > 1.30:
            return ratio, pct_red
        elif ratio < 0.70:
            return ratio, pct_org
        return ratio, pct_grn

    def _pct_cell(pred_n: int, total_pred: int):
        """Pred distribution share (no GT): fraction → shown as %."""
        if total_pred == 0 or pred_n == 0:
            return "—", pct_grey
        return pred_n / total_pred, pct_fmt

    def _write_val(row, col, val, fmt):
        if isinstance(val, float):
            ws.write_number(row, col, val, fmt)
        else:
            ws.write(row, col, val, fmt)

    # ── Block writer ──────────────────────────────────────────────────────────

    def _write_grid_block(start_row, title, pred_grid, gt_grid=None):
        """
        Writes three side-by-side 3×3 tables per block:
          Col 0–3:   Predictions (counts)
          Col 5–8:   Ground Truth counts  OR  Det. Share %  (if no GT)
          Col 10–13: Coverage Ratio %      OR  same Det. Share % (redundant label)
        """
        ws.write(start_row, 0, title, bold_fmt)
        start_row += 1

        total_pred = sum(pred_grid[r][c] for r in range(3) for c in range(3))
        total_gt   = sum(gt_grid[r][c]   for r in range(3) for c in range(3)) if gt_grid else 0

        # Headers
        ws.write(start_row, 0,  "Predictions →",  hdr_fmt)
        ws.write(start_row, 5,  "Ground Truth →" if gt_grid else "Det. Share % →", hdr_fmt)
        ws.write(start_row, 10, "Coverage Ratio % →" if gt_grid else "Det. Share % →", hdr_fmt)
        if gt_grid:
            ws.write(start_row, 13,
                     ">130% = over-detect  |  <70% = miss  |  70–130% = ok",
                     note_fmt)
        for ci, cl in enumerate(col_labels):
            ws.write(start_row, ci + 1,  cl, hdr_fmt)
            ws.write(start_row, ci + 6,  cl, hdr_fmt)
            ws.write(start_row, ci + 11, cl, hdr_fmt)
        start_row += 1

        for ri, rl in enumerate(row_labels):
            r = start_row + ri
            # ── Predictions (counts) ──────────────────────────────────────────
            ws.write(r, 0, rl, bold_fmt)
            for ci in range(3):
                ws.write(r, ci + 1, pred_grid[ri][ci], ctr_fmt)

            # ── GT counts or Det. Share % ─────────────────────────────────────
            ws.write(r, 5, rl, bold_fmt)
            if gt_grid:
                for ci in range(3):
                    ws.write(r, ci + 6, gt_grid[ri][ci], ctr_fmt)
            else:
                for ci in range(3):
                    v, fmt = _pct_cell(pred_grid[ri][ci], total_pred)
                    _write_val(r, ci + 6, v, fmt)

            # ── Coverage Ratio % (or Det. Share %) ───────────────────────────
            ws.write(r, 10, rl, bold_fmt)
            if gt_grid:
                for ci in range(3):
                    v, fmt = _ratio_cell(pred_grid[ri][ci], gt_grid[ri][ci])
                    _write_val(r, ci + 11, v, fmt)
            else:
                for ci in range(3):
                    v, fmt = _pct_cell(pred_grid[ri][ci], total_pred)
                    _write_val(r, ci + 11, v, fmt)

        # Color scales on the raw-count grids
        pr = f"B{start_row+1}:D{start_row+3}"
        ws.conditional_format(pr, {"type": "2_color_scale",
                                   "min_color": _C_WHITE, "max_color": _C_DKBLUE})
        if gt_grid:
            gr = f"G{start_row+1}:I{start_row+3}"
            ws.conditional_format(gr, {"type": "2_color_scale",
                                       "min_color": _C_WHITE, "max_color": _C_GREEN})

        return start_row + 3 + 2   # 3 data rows + 2 blank separator rows

    # ── Collect overall data and write ────────────────────────────────────────

    all_dets = [det for r in predict_results for det in r["detections"]]
    overall_pred = _build_grid(all_dets)
    all_gt_boxes = (
        [g for r in predict_results
         for g in [p["gt"] for p in r.get("tp_pairs", [])] + r.get("fn_gts", [])]
        if has_gt else None
    )
    overall_gt = _build_gt_grid(all_gt_boxes) if all_gt_boxes else None

    current_row = _write_grid_block(0, "OVERALL — All Classes", overall_pred, overall_gt)

    for cls in all_classes:
        cls_pred_dets = [det for r in predict_results for det in r["detections"]
                         if det["class_name"] == cls]
        if not cls_pred_dets and not has_gt:
            continue
        cls_pred_grid = _build_grid(cls_pred_dets)

        cls_gt_grid = None
        if has_gt:
            cls_gt_boxes = []
            for r in predict_results:
                cls_gt_boxes += [p["gt"] for p in r.get("tp_pairs", []) if p["gt"]["class_name"] == cls]
                cls_gt_boxes += [g for g in r.get("fn_gts", []) if g["class_name"] == cls]
            if cls_gt_boxes:
                cls_gt_grid = _build_gt_grid(cls_gt_boxes)

        current_row = _write_grid_block(current_row, f"Class: {cls}", cls_pred_grid, cls_gt_grid)

    # Column widths
    ws.set_column(0,  0,  22)   # pred row label / title
    ws.set_column(1,  3,  10)   # pred counts
    ws.set_column(4,  4,   3)   # spacer
    ws.set_column(5,  5,  22)   # gt/pct label
    ws.set_column(6,  8,  10)   # gt counts / pct
    ws.set_column(9,  9,   3)   # spacer
    ws.set_column(10, 10, 22)   # ratio label
    ws.set_column(11, 13, 10)   # ratio values
    ws.set_column(14, 14, 50)   # legend note
