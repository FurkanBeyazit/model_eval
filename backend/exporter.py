"""
exporter.py  –  Excel / CSV Export
Produces a multi-sheet Excel workbook from validation & prediction results.

Sheets:
  Summary            – overall metrics + run stats
  Val_Class_Metrics  – per-class P / R / F1 / mAP from model.val()
  Per_Image          – detection counts + avg/max confidence per class per image
  All_Detections     – every single bounding box (image, class, conf, x1, y1, x2, y2, w, h, area)
  Class_Distribution – aggregated counts + confidence stats per class
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
from datetime import datetime


def export_to_excel(
    predict_results: List[dict],
    val_metrics: Optional[dict],
    class_names: Dict[int, str],
    output_path: Optional[str] = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = str(
            Path(tempfile.gettempdir()) / f"model_eval_{timestamp}.xlsx"
        )

    all_classes = sorted(set(class_names.values()))
    total_images = len(predict_results)
    total_dets = sum(r["total_detections"] for r in predict_results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # ── Sheet 1: Summary ─────────────────────────────────────────────────────
        summary_rows = [
            {"Metric": "Export Timestamp", "Value": timestamp},
            {"Metric": "Total Images Analyzed", "Value": total_images},
            {"Metric": "Total Detections", "Value": total_dets},
        ]
        if val_metrics:
            summary_rows += [
                {"Metric": "mAP50",          "Value": val_metrics.get("map50", "-")},
                {"Metric": "mAP50-95",       "Value": val_metrics.get("map50_95", "-")},
                {"Metric": "Mean Precision",  "Value": val_metrics.get("mp", "-")},
                {"Metric": "Mean Recall",     "Value": val_metrics.get("mr", "-")},
            ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # ── Sheet 2: Val_Class_Metrics ────────────────────────────────────────────
        if val_metrics and val_metrics.get("class_metrics"):
            pd.DataFrame(val_metrics["class_metrics"]).to_excel(
                writer, sheet_name="Val_Class_Metrics", index=False
            )

        # ── Sheet 3: Per_Image ───────────────────────────────────────────────────
        if predict_results:
            rows = []
            for r in predict_results:
                row: dict = {
                    "Image":       r["image_name"],
                    "Image_Path":  r["image_path"],
                    "Total":       r["total_detections"],
                }
                for cls in all_classes:
                    row[f"{cls}_count"]    = r["class_counts"].get(cls, 0)
                    avg = r["class_avg_conf"].get(cls)
                    mx  = r["class_max_conf"].get(cls)
                    row[f"{cls}_avg_conf"] = avg if avg is not None else ""
                    row[f"{cls}_max_conf"] = mx  if mx  is not None else ""
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name="Per_Image", index=False)

        # ── Sheet 4: All_Detections ───────────────────────────────────────────────
        if predict_results:
            all_dets = []
            for r in predict_results:
                for det in r["detections"]:
                    w = det["x2"] - det["x1"]
                    h = det["y2"] - det["y1"]
                    all_dets.append(
                        {
                            "image":      r["image_name"],
                            "image_path": r["image_path"],
                            "class":      det["class_name"],
                            "class_id":   det["class_id"],
                            "confidence": det["confidence"],
                            "x1": det["x1"], "y1": det["y1"],
                            "x2": det["x2"], "y2": det["y2"],
                            "width":  round(w, 1),
                            "height": round(h, 1),
                            "area":   round(w * h, 1),
                        }
                    )
            if all_dets:
                pd.DataFrame(all_dets).to_excel(
                    writer, sheet_name="All_Detections", index=False
                )

        # ── Sheet 5: Class_Distribution ───────────────────────────────────────────
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
                dist_rows.append(
                    {
                        "Class":            cls,
                        "Total_Detections": len(confs),
                        "Images_With_Det":  sum(
                            1 for r in predict_results if cls in r["class_counts"]
                        ),
                        "Avg_Confidence":   round(sum(confs) / len(confs), 4) if confs else 0,
                        "Min_Confidence":   round(min(confs), 4) if confs else 0,
                        "Max_Confidence":   round(max(confs), 4) if confs else 0,
                    }
                )
            pd.DataFrame(dist_rows).to_excel(
                writer, sheet_name="Class_Distribution", index=False
            )

    return output_path
