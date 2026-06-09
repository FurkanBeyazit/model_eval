"""
gradio_app.py  –  Danusys Model Eval Platform — Gradio Frontend
"""

import os
import io
import tempfile
import tkinter as tk
from tkinter import filedialog

import gradio as gr
import pandas as pd
import requests
from PIL import Image

API_BASE = os.environ.get("API_BASE", "http://localhost:8001")
TIMEOUT  = 600


# ── API helpers ───────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> dict:
    resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _get(endpoint: str, params=None):
    resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


def _check_backend() -> bool:
    try:
        return _get("/health").status_code == 200
    except Exception:
        return False


def _fetch_annotated(image_name: str, classes: str = "", highlight_idx: int = -1):
    """Fetch annotated JPEG from backend, returns PIL Image or None."""
    try:
        resp = requests.get(
            f"{API_BASE}/api/analysis/image/annotated",
            params={"image_name": image_name, "classes": classes,
                    "highlight_idx": highlight_idx},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    except Exception:
        return None


def _fetch_annotated_custom(image_path: str, detections: list, highlight_idx: int = -1):
    """Annotate image using explicit detections list — does NOT depend on backend state."""
    try:
        resp = requests.post(
            f"{API_BASE}/api/analysis/image/annotate_custom",
            json={"image_path": image_path, "detections": detections,
                  "highlight_idx": highlight_idx},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    except Exception:
        return None


def _build_det_df(dets: list) -> pd.DataFrame:
    if not dets:
        return pd.DataFrame(columns=["#", "Class", "Conf", "X1", "Y1", "X2", "Y2", "W", "H"])
    return pd.DataFrame([
        {"#": i, "Class": d["class_name"], "Conf": d["confidence"],
         "X1": int(d["x1"]), "Y1": int(d["y1"]),
         "X2": int(d["x2"]), "Y2": int(d["y2"]),
         "W": int(d["x2"] - d["x1"]), "H": int(d["y2"] - d["y1"])}
        for i, d in enumerate(dets)
    ])


# ── Upload callbacks ──────────────────────────────────────────────────────────

def _file_path(file_obj) -> str:
    """Normalize Gradio file object → local path string."""
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("name", "")
    return getattr(file_obj, "name", str(file_obj))


def model_upload_cb(file_obj):
    """User uploaded a .pt file — its Gradio temp path is directly usable by backend."""
    return _file_path(file_obj)


def dataset_zip_upload_cb(file_obj):
    """User uploaded a dataset .zip — backend extracts it and returns the folder path."""
    path = _file_path(file_obj)
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                f"{API_BASE}/api/upload/dataset",
                files={"file": (os.path.basename(path), f, "application/zip")},
                timeout=300,
            )
        resp.raise_for_status()
        return resp.json()["dataset_path"]
    except Exception as exc:
        return f"Upload error: {exc}"


# ── Browse dialogs ────────────────────────────────────────────────────────────

def browse_model_cb():
    root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", 1)
    path = filedialog.askopenfilename(
        title="Select Model (.pt)", filetypes=[("YOLO Model", "*.pt"), ("All files", "*.*")]
    )
    root.destroy(); return path or ""


def browse_dataset_cb():
    root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", 1)
    path = filedialog.askdirectory(title="Select Dataset Folder")
    root.destroy(); return path or ""


# ── DataFrame builders ────────────────────────────────────────────────────────

def _per_image_df(results: list, has_gt: bool = False) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    all_classes = sorted({det["class_name"] for r in results for det in r["detections"]})
    rows = []
    for r in results:
        row = {"Image": r["image_name"], "Total_Pred": r["total_detections"]}
        if has_gt:
            row["Total_GT"]  = sum(r.get("gt_counts", {}).values())
            row["TP"]        = r.get("tp", "")
            row["FP"]        = r.get("fp", "")
            row["FN_Missed"] = r.get("fn", "")
            mr = r.get("match_rate")
            row["Match_Rate"] = f"{mr:.3f}" if mr is not None else ""
        for cls in all_classes:
            row[cls] = r["class_counts"].get(cls, 0)
            if has_gt:
                row[f"{cls}_gt"] = r.get("gt_counts", {}).get(cls, 0)
            conf = r["class_avg_conf"].get(cls)
            row[f"{cls}_conf"] = f"{conf:.3f}" if conf is not None else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _worst_images_df(results: list, has_gt: bool = False) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        confs = [det["confidence"] for det in r["detections"]]
        row = {"Image": r["image_name"], "Total_Pred": r["total_detections"],
               "Avg_Conf": round(sum(confs) / len(confs), 3) if confs else 0.0}
        if has_gt:
            row["Total_GT"]  = sum(r.get("gt_counts", {}).values())
            row["FN_Missed"] = r.get("fn", 0)
            row["FP_Extra"]  = r.get("fp", 0)
            row["TP"]        = r.get("tp", 0)
            mr = r.get("match_rate")
            row["Match_Rate"] = f"{mr:.3f}" if mr is not None else "N/A"
        else:
            row["Zero_Det"] = r["total_detections"] == 0
        rows.append(row)
    df = pd.DataFrame(rows)
    if has_gt:
        df = df.sort_values(["FN_Missed", "Avg_Conf"], ascending=[False, True])
    else:
        df = df.sort_values(["Zero_Det", "Avg_Conf"], ascending=[False, True])
    return df.head(20).reset_index(drop=True)


def _val_combined_df(val_data: dict, overall_p=None, overall_r=None) -> pd.DataFrame:
    """Build the combined validation table (matches YOLO console output column order).
    overall_p / overall_r: GT-matching based values — replace mp/mr in the 'all' row.
    """
    p = overall_p if overall_p is not None else val_data.get("mp", 0)
    r = overall_r if overall_r is not None else val_data.get("mr", 0)
    f1_all = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0
    rows = [{
        "Class": "all",
        "Images":     val_data.get("total_images", ""),
        "Instances":  val_data.get("total_instances", ""),
        "Precision": p, "Recall": r, "F1": f1_all,
        "mAP50": val_data.get("map50", ""),
        "mAP50-95": val_data.get("map50_95", ""),
    }]
    for m in val_data.get("class_metrics", []):
        rows.append({
            "Class":     m["class"],
            "Images":    m.get("Images", ""),
            "Instances": m.get("Instances", ""),
            "Precision": m["Precision"], "Recall": m["Recall"], "F1": m["F1"],
            "mAP50": m["mAP50"], "mAP50-95": m["mAP50-95"],
        })
    return pd.DataFrame(rows)


def _history_df() -> pd.DataFrame:
    try:
        data = _get("/api/analysis/history").json()
        rows = data.get("history", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "mAP50" in df.columns:
            df = df.sort_values("mAP50", ascending=False, na_position="last")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── Model callbacks ───────────────────────────────────────────────────────────

def load_model_cb(model_path: str):
    if not model_path or not model_path.strip():
        return "No path specified.", ""
    try:
        data = _post("/api/model/load", {"model_path": model_path})
        if data["ok"]:
            classes_lines = "\n".join(
                f"  [{k}] {v}"
                for k, v in sorted(data["classes"].items(), key=lambda x: int(x[0]))
            )
            return "✓  Model Ready", classes_lines
        return f"✗  {data['message']}", ""
    except Exception as exc:
        return f"✗  Backend error: {exc}", ""


# ── Run callbacks ─────────────────────────────────────────────────────────────

def run_val_cb(dataset_path: str, conf: float, iou: float):
    if not _check_backend():
        return "Backend not running — start backend/main.py first.", {}, None
    try:
        data = _post("/api/analysis/validate",
                     {"dataset_path": dataset_path, "conf": conf, "iou": iou})
        return "Validation complete.", data, _val_combined_df(data)
    except Exception as exc:
        return f"Error: {exc}", {}, None


def run_predict_cb(dataset_path: str, conf: float, iou_thresh: float):
    if not _check_backend():
        return "Backend not running.", [], False, None, gr.update(choices=[], value=None), None
    try:
        data = _post("/api/analysis/predict",
                     {"dataset_path": dataset_path, "conf": conf, "iou_thresh": iou_thresh})
        results = data["results"]
        has_gt  = data.get("has_gt", False)
        names   = [r["image_name"] for r in results]
        gt_note = " | GT labels found ✓" if has_gt else " | No GT labels"
        return (
            f"{data['total_images']} images | {data['total_detections']} detections{gt_note}",
            results, has_gt,
            _per_image_df(results, has_gt),
            gr.update(choices=names, value=names[0] if names else None),
            _worst_images_df(results, has_gt),
        )
    except Exception as exc:
        return f"Error: {exc}", [], False, None, gr.update(choices=[], value=None), None


def run_both_cb(dataset_path: str, conf: float, iou: float, iou_thresh: float):
    if not _check_backend():
        empty = gr.update(choices=[], value=None)
        return "Backend not running.", {}, None, [], False, None, empty, None
    try:
        data = _post("/api/analysis/both", {
            "dataset_path": dataset_path, "conf": conf,
            "iou": iou, "iou_thresh": iou_thresh,
        })
        val     = data["val_metrics"]
        pred    = data["predict"]
        results = pred["results"]
        has_gt  = pred.get("has_gt", False)
        names   = [r["image_name"] for r in results]
        gt_note = " | GT labels found ✓" if has_gt else " | No GT labels"
        status  = (f"Val mAP50={val['map50']} | "
                   f"Predict: {pred['total_images']} images, "
                   f"{pred['total_detections']} detections{gt_note}")

        # Use GT-matching P/R for the "all" row so it matches Run History
        overall_p = overall_r = None
        if has_gt:
            total_tp = sum(r.get("tp", 0) for r in results)
            total_fp = sum(r.get("fp", 0) for r in results)
            total_fn = sum(r.get("fn", 0) for r in results)
            overall_p = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else None
            overall_r = round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else None

        return (
            status,
            val,
            _val_combined_df(val, overall_p, overall_r),
            results, has_gt,
            _per_image_df(results, has_gt),
            gr.update(choices=names, value=names[0] if names else None),
            _worst_images_df(results, has_gt),
        )
    except Exception as exc:
        empty = gr.update(choices=[], value=None)
        return f"Error: {exc}", {}, None, [], False, None, empty, None


# ── Image viewer callbacks ────────────────────────────────────────────────────

def image_load_cb(image_name: str, predict_state: list):
    """Called when a new image is selected — resets class filter, shows all detections."""
    if not image_name or not predict_state:
        return None, pd.DataFrame(), gr.update(choices=[], value=[])
    result = next((r for r in predict_state if r["image_name"] == image_name), None)
    if not result:
        return None, pd.DataFrame(), gr.update(choices=[], value=[])
    all_classes = sorted(set(d["class_name"] for d in result["detections"]))
    pil_img     = _fetch_annotated(image_name)
    det_df      = _build_det_df(result["detections"])
    return pil_img, det_df, gr.update(choices=all_classes, value=[])


def image_filter_cb(image_name: str, predict_state: list, class_filter: list):
    """Called when class filter changes — keeps same image, applies filter."""
    if not image_name or not predict_state:
        return None, pd.DataFrame()
    result = next((r for r in predict_state if r["image_name"] == image_name), None)
    if not result:
        return None, pd.DataFrame()
    classes_param = ",".join(class_filter) if class_filter else ""
    pil_img = _fetch_annotated(image_name, classes=classes_param)
    dets    = [d for d in result["detections"]
               if not class_filter or d["class_name"] in class_filter]
    return pil_img, _build_det_df(dets)


def on_det_select(evt: gr.SelectData, image_name: str, predict_state: list,
                  class_filter: list):
    """Highlight the clicked detection row on the image."""
    if not image_name or not predict_state:
        return None
    row_idx = evt.index[0]
    classes_param = ",".join(class_filter) if class_filter else ""
    return _fetch_annotated(image_name, classes=classes_param, highlight_idx=row_idx)


def view_comparison_cb(image_name: str, predict_state: list, has_gt: bool):
    if not image_name or not predict_state or not has_gt:
        return None, pd.DataFrame()
    pil_img = None
    try:
        resp    = _get("/api/analysis/image/comparison", {"image_name": image_name})
        pil_img = Image.open(io.BytesIO(resp.content))
    except Exception:
        pass
    result = next((r for r in predict_state if r["image_name"] == image_name), None)
    if result is None:
        return pil_img, pd.DataFrame()
    rows = []
    for pair in result.get("tp_pairs", []):
        rows.append({"Type": "TP", "Class": pair["pred"]["class_name"],
                     "Conf": round(pair["pred"]["confidence"], 3),
                     "IoU": round(pair["iou"], 3)})
    for p in result.get("fp_preds", []):
        rows.append({"Type": "FP", "Class": p["class_name"],
                     "Conf": round(p["confidence"], 3), "IoU": ""})
    for g in result.get("fn_gts", []):
        rows.append({"Type": "FN", "Class": g["class_name"], "Conf": "", "IoU": ""})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Type", "Class", "Conf", "IoU"])
    return pil_img, df


# ── Model Comparison helpers ──────────────────────────────────────────────────

def _run_one_model_for_compare(model_path, dataset_path,
                                conf, iou, iou_thresh, label):
    """Load a model, run both, return E02-filtered (first 33) results."""
    if not model_path or not model_path.strip():
        return None
    model_name = os.path.basename(model_path.replace("\\", "/"))

    load_resp = _post("/api/model/load", {"model_path": model_path})
    if not load_resp.get("ok"):
        return {"label": label, "model_name": model_name,
                "error": load_resp.get("message", "Load failed"),
                "val_metrics": {}, "results": [], "has_gt": False}
    try:
        data = _post("/api/analysis/both", {
            "dataset_path": dataset_path,
            "conf": conf, "iou": iou, "iou_thresh": iou_thresh,
        })
    except Exception as exc:
        return {"label": label, "model_name": model_name,
                "error": str(exc),
                "val_metrics": {}, "results": [], "has_gt": False}

    all_results = data["predict"]["results"]
    e02 = [r for r in all_results
           if r["image_name"].upper().startswith("E02")][:33]
    return {
        "label":       label,
        "model_name":  model_name,
        "error":       None,
        "val_metrics": data["val_metrics"],
        "results":     e02,
        "has_gt":      data["predict"].get("has_gt", False),
    }


def _cmp_metrics_md(data) -> str:
    if data is None:
        return "_Not configured_"
    if data.get("error"):
        return f"**{data['model_name']}**\n\n⚠ {data['error']}"
    v  = data["val_metrics"]
    rs = data["results"]
    total_dets = sum(r["total_detections"] for r in rs)
    lines = [
        f"### {data['model_name']}",
        f"**mAP50:** {v.get('map50', '—')} &nbsp;|&nbsp; **mAP50-95:** {v.get('map50_95', '—')}",
        f"**P:** {v.get('mp', '—')} &nbsp;|&nbsp; **R:** {v.get('mr', '—')}",
        f"E02 images: **{len(rs)}** &nbsp;|&nbsp; Detections: **{total_dets}**",
    ]
    if data["has_gt"] and rs:
        tp = sum(r.get("tp", 0) for r in rs)
        fp = sum(r.get("fp", 0) for r in rs)
        fn = sum(r.get("fn", 0) for r in rs)
        p  = round(tp / (tp + fp), 3) if (tp + fp) > 0 else "—"
        r_ = round(tp / (tp + fn), 3) if (tp + fn) > 0 else "—"
        lines.append(f"TP: {tp} &nbsp;|&nbsp; FP: {fp} &nbsp;|&nbsp; FN: {fn}")
        lines.append(f"GT-P: {p} &nbsp;|&nbsp; GT-R: {r_}")
    return "\n\n".join(lines)


def _cmp_table(models_data: list) -> pd.DataFrame:
    """Per-image falldown catch table: GT | Model1_caught | Model2_caught | ...
    Bottom rows show totals and catch rate (%).
    If no GT labels, shows raw prediction counts instead.
    """
    valid = [d for d in models_data if d is not None and not d.get("error")]
    if not valid:
        return pd.DataFrame()

    has_gt = any(d["has_gt"] for d in valid)

    # Ordered unique E02 image names
    all_names = []
    seen = set()
    for d in valid:
        for r in d["results"]:
            if r["image_name"] not in seen:
                all_names.append(r["image_name"])
                seen.add(r["image_name"])

    model_totals = {d["model_name"]: 0 for d in valid}
    gt_total = 0

    rows = []
    for name in all_names:
        row = {"Image": name}

        # GT column (same value regardless of model — use first valid model with GT)
        if has_gt:
            gt_val = ""
            for d in valid:
                r = next((x for x in d["results"] if x["image_name"] == name), None)
                if r:
                    gt_val = r.get("gt_counts", {}).get("falldown", 0)
                    break
            row["GT_Fall"] = gt_val
            if isinstance(gt_val, int):
                gt_total += gt_val

        for d in valid:
            r = next((x for x in d["results"] if x["image_name"] == name), None)
            mn = d["model_name"]
            if r is None:
                if has_gt:
                    row[f"{mn}_TP"] = ""
                    row[f"{mn}_FP"] = ""
                else:
                    row[f"{mn}_pred"] = ""
                continue
            if has_gt:
                tp = sum(1 for p in r.get("tp_pairs", [])
                         if p["pred"]["class_name"] == "falldown")
                fp = sum(1 for p in r.get("fp_preds", [])
                         if p["class_name"] == "falldown")
                row[f"{mn}_TP"] = tp
                row[f"{mn}_FP"] = fp
                model_totals[f"{mn}_TP"] = model_totals.get(f"{mn}_TP", 0) + tp
                model_totals[f"{mn}_FP"] = model_totals.get(f"{mn}_FP", 0) + fp
            else:
                pred = r["class_counts"].get("falldown", 0)
                row[f"{mn}_pred"] = pred
                model_totals[f"{mn}_pred"] = model_totals.get(f"{mn}_pred", 0) + pred

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Summary rows ──────────────────────────────────────────────────────────
    total_row = {"Image": "── TOPLAM ──"}
    pct_row   = {"Image": "── YAKALAMA % ──"}
    if has_gt:
        total_row["GT_Fall"] = gt_total
        pct_row["GT_Fall"]   = "100%"
    for d in valid:
        mn = d["model_name"]
        if has_gt:
            tp_t = model_totals.get(f"{mn}_TP", 0)
            fp_t = model_totals.get(f"{mn}_FP", 0)
            recall = round(tp_t / gt_total * 100, 1) if gt_total > 0 else 0
            total_row[f"{mn}_TP"] = tp_t
            total_row[f"{mn}_FP"] = fp_t
            pct_row[f"{mn}_TP"] = f"Recall {recall}%  ({tp_t}/{gt_total})"
            pct_row[f"{mn}_FP"] = f"{fp_t} yanlış alarm"
        else:
            t = model_totals.get(f"{mn}_pred", 0)
            total_row[f"{mn}_pred"] = t
            pct_row[f"{mn}_pred"]   = "—"

    df = pd.concat([df, pd.DataFrame([total_row, pct_row])], ignore_index=True)
    return df


def run_compare_cb(m1: str, m2: str, m3: str,
                   dataset_path: str, conf: float,
                   iou: float, iou_thresh: float):
    """Run up to 3 models sequentially and return comparison results."""
    if not dataset_path.strip():
        _empty = gr.update(choices=[], value=None)
        return ("Dataset path required.",
                [None, None, None],
                "_", "_", "_", pd.DataFrame(), _empty)

    specs = [(m1, "Model 1"), (m2, "Model 2"), (m3, "Model 3")]
    models_data = []
    for mp, label in specs:
        models_data.append(
            _run_one_model_for_compare(mp, dataset_path, conf, iou, iou_thresh, label)
        )

    valid = [d for d in models_data if d is not None]
    if not valid:
        _empty = gr.update(choices=[], value=None)
        return ("No valid models provided.",
                models_data,
                "_", "_", "_", pd.DataFrame(), _empty)

    md1 = _cmp_metrics_md(models_data[0])
    md2 = _cmp_metrics_md(models_data[1])
    md3 = _cmp_metrics_md(models_data[2])
    cdf = _cmp_table(models_data)

    all_names = []
    seen = set()
    for d in valid:
        for r in d["results"]:
            if r["image_name"] not in seen:
                all_names.append(r["image_name"])
                seen.add(r["image_name"])

    errors = [d["model_name"] for d in valid if d.get("error")]
    status = (
        f"Done — {len(valid)} model(s) | {len(all_names)} E02 images"
        + (f" | Errors: {', '.join(errors)}" if errors else "")
    )
    return (status, models_data, md1, md2, md3, cdf,
            gr.update(choices=all_names, value=all_names[0] if all_names else None))


def compare_image_cb(image_name: str, models_data: list):
    """Fetch annotated images for the selected image from all 3 model result sets."""
    imgs = [None, None, None]
    if not image_name or not models_data:
        return imgs[0], imgs[1], imgs[2]
    for i, d in enumerate(models_data):
        if d is None or d.get("error"):
            continue
        r = next((x for x in d["results"] if x["image_name"] == image_name), None)
        if r is None:
            continue
        imgs[i] = _fetch_annotated_custom(r["image_path"], r["detections"])
    return imgs[0], imgs[1], imgs[2]


# ── Export / history ──────────────────────────────────────────────────────────

def export_cb():
    try:
        resp = requests.post(f"{API_BASE}/api/export/excel", timeout=120)
        if resp.status_code != 200:
            return f"Error: {resp.json().get('detail', 'Unknown error')}", None
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.write(resp.content); tmp.close()
        return "Excel ready to download.", tmp.name
    except Exception as exc:
        return f"Error: {exc}", None


def export_history_csv_cb():
    try:
        data = _get("/api/analysis/history").json()
        rows = data.get("history", [])
        if not rows:
            return "No history to export.", None
        df = pd.DataFrame(rows)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df.to_csv(tmp.name, index=False, encoding="utf-8-sig")
        tmp.close()
        return "CSV ready.", tmp.name
    except Exception as exc:
        return f"Error: {exc}", None


def clear_history_cb():
    try:
        requests.delete(f"{API_BASE}/api/analysis/history", timeout=10)
    except Exception:
        pass
    return pd.DataFrame()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")
CSS   = """
footer { display: none !important; }
.model-status textarea { font-size: 1.1em; font-weight: bold; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Danusys Model Eval Platform", css=CSS, theme=THEME) as demo:

        gr.Markdown("# Danusys Model Eval Platform\nYOLO fine-tuned model evaluation")

        val_state     = gr.State({})
        predict_state = gr.State([])
        has_gt_state  = gr.State(False)

        with gr.Row(equal_height=False):

            # ── Left: Config ──────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):

                gr.Markdown("### Model")
                with gr.Row():
                    model_path_inp   = gr.Textbox(label="Model path (.pt)",
                                                  placeholder=r"D:\models\best.pt",
                                                  lines=1, scale=4, interactive=True)
                    btn_browse_model = gr.Button("Browse", scale=1, min_width=60)
                with gr.Accordion("Upload model from your PC", open=False):
                    model_upload_file = gr.File(
                        label="Select .pt file",
                        file_types=[".pt"],
                        file_count="single",
                    )
                btn_load     = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Status", interactive=False, lines=1,
                                          elem_classes=["model-status"])
                with gr.Accordion("Classes", open=False):
                    classes_box = gr.Textbox(interactive=False, lines=8,
                                             max_lines=20, show_label=False)

                gr.Markdown("### Dataset")
                with gr.Row():
                    dataset_inp        = gr.Textbox(
                        label="Dataset folder",
                        placeholder=r"C:\datasets\cctv_test",
                        info="Supported: root/val/images · root/images · root/",
                        lines=1, scale=4, interactive=True)
                    btn_browse_dataset = gr.Button("Browse", scale=1, min_width=60)
                with gr.Accordion("Upload dataset from your PC", open=False):
                    gr.Markdown(
                        "_Zip your dataset folder (images + labels) and upload here. "
                        "Windows: sağ tık → Sıkıştır (.zip)_",
                        elem_id="zip-hint",
                    )
                    dataset_upload_file = gr.File(
                        label="Select dataset .zip",
                        file_types=[".zip"],
                        file_count="single",
                    )

                conf_slider = gr.Slider(
                    0.05, 0.95, value=0.50, step=0.05,
                    label="Confidence threshold  (fixed)",
                    info="Only keep predictions with score ≥ this value.",
                    interactive=False)

                iou_slider = gr.Slider(
                    0.10, 0.95, value=0.45, step=0.05,
                    label="IoU threshold  (Validation only)",
                    info="Used by model.val() internally. Does not affect GT matching.")

                iou_thresh_slider = gr.Slider(
                    0.10, 0.90, value=0.50, step=0.05,
                    label="GT match IoU  (TP / FP / FN)",
                    info="Spatial overlap needed to call a detection a True Positive. "
                         "Unrelated to confidence — two high-confidence boxes can still "
                         "be FP if they don't overlap the GT box enough.")

                gr.Markdown("### Run")
                with gr.Row():
                    btn_val     = gr.Button("Validation",  variant="secondary", size="sm")
                    btn_predict = gr.Button("Per-Image",   variant="secondary", size="sm")
                btn_both   = gr.Button("Run Both", variant="primary")
                run_status = gr.Textbox(label="Status", interactive=False, lines=2)

            # ── Right: Results ────────────────────────────────────────────────
            with gr.Column(scale=3):
                with gr.Tabs():

                    # ── Validation ────────────────────────────────────────────
                    with gr.Tab("Validation Metrics"):
                        gr.Markdown("Aggregate metrics from `model.val()` — requires GT label files.")
                        val_class_df = gr.DataFrame(interactive=False, wrap=True)

                    # ── Image Viewer ──────────────────────────────────────────
                    with gr.Tab("Image Viewer"):
                        with gr.Row():
                            img_dropdown = gr.Dropdown(
                                label="Image", choices=[], interactive=True, scale=3)
                            class_filter_dd = gr.Dropdown(
                                label="Filter classes (empty = show all)",
                                choices=[], multiselect=True, interactive=True, scale=2)
                            btn_view = gr.Button("Refresh", variant="secondary",
                                                 scale=1, min_width=80)

                        with gr.Tabs():

                            with gr.Tab("Prediction"):
                                annotated_img = gr.Image(
                                    label="Annotated image", type="pil",
                                    height=600, show_download_button=True)
                                gr.Markdown(
                                    "_Click a row in the table below to highlight "
                                    "that detection on the image._")
                                det_df = gr.DataFrame(
                                    label="Detections  (click row to highlight)",
                                    interactive=False, wrap=True)

                            with gr.Tab("GT Comparison"):
                                gr.HTML("""
<div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;
            padding:9px 12px;background:#1e1e2e;border-radius:8px;margin-bottom:8px">
  <span style="display:flex;align-items:center;gap:7px;color:#e0e0e0;font-size:0.9em">
    <span style="width:16px;height:16px;background:#00c853;border:2px solid #00c853;
                 display:inline-block;border-radius:2px;flex-shrink:0"></span>
    <b>TP</b> — Correct detection
  </span>
  <span style="display:flex;align-items:center;gap:7px;color:#e0e0e0;font-size:0.9em">
    <span style="width:16px;height:16px;background:#f44336;border:2px solid #f44336;
                 display:inline-block;border-radius:2px;flex-shrink:0"></span>
    <b>FP</b> — Extra detection, no GT match
  </span>
  <span style="display:flex;align-items:center;gap:7px;color:#e0e0e0;font-size:0.9em">
    <span style="width:16px;height:16px;background:#ff9800;border:2px solid #ff9800;
                 display:inline-block;border-radius:2px;flex-shrink:0"></span>
    <b>FN</b> — Missed object
  </span>
</div>
<p style="color:#888;font-size:0.82em;margin:4px 0 0 4px">
  Requires GT label (.txt) files in the dataset.
</p>
""")
                                btn_compare = gr.Button("Show GT Comparison",
                                                        variant="primary")
                                comparison_img = gr.Image(
                                    label="GT vs Prediction", type="pil",
                                    height=600, show_download_button=True)
                                compare_df = gr.DataFrame(
                                    label="TP / FP / FN Detail",
                                    interactive=False, wrap=True)

                    # ── Worst Images ──────────────────────────────────────────
                    with gr.Tab("Worst Images"):
                        gr.Markdown(
                            "GT available → sorted by most missed (FN).  \n"
                            "No GT → zero-detection + lowest confidence.  _(Top 20)_")
                        worst_df_out = gr.DataFrame(interactive=False, wrap=True)

                    # ── Run History ───────────────────────────────────────────
                    with gr.Tab("Run History"):
                        gr.Markdown(
                            "Every run is logged here.  "
                            "Persists across sessions (`~/.model_eval_history.json`).")
                        with gr.Row():
                            btn_refresh_history = gr.Button("Refresh", variant="secondary",
                                                            size="sm")
                            btn_export_history  = gr.Button("Export CSV", variant="secondary",
                                                            size="sm")
                            btn_clear_history   = gr.Button("Clear History", variant="stop",
                                                            size="sm")
                        history_export_status = gr.Textbox(label="", interactive=False,
                                                           lines=1, visible=False)
                        history_export_file   = gr.File(label="Download CSV",
                                                        interactive=False, visible=False)
                        history_df = gr.DataFrame(interactive=False, wrap=True)

                    # ── Export ────────────────────────────────────────────────
                    with gr.Tab("Export"):
                        gr.Markdown(
                            "**Excel sheets:** Summary · Val_Class_Metrics · "
                            "Class_Performance · Per_Image · Per_Image_GT · "
                            "All_Detections · Class_Distribution · Confusion_Matrix · "
                            "Threshold_Curve · Size_Analysis · Worst_Images · Spatial_Bias")
                        btn_export    = gr.Button("Generate Excel",
                                                  variant="primary", size="lg")
                        export_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        export_file   = gr.File(label="Download", interactive=False)

                    # ── Per-Image (raw data, rarely needed) ───────────────────
                    with gr.Tab("Per-Image Results"):
                        gr.Markdown(
                            "Detection counts per image. "
                            "GT columns appear automatically when label files are found.")
                        per_image_df = gr.DataFrame(interactive=False, wrap=True)

                    # ── Model Comparison ──────────────────────────────────────
                    with gr.Tab("Model Comparison"):
                        gr.Markdown(
                            "## Model Comparison\n"
                            "Run up to **3 models** on the same dataset side-by-side.  \n"
                            "Results are filtered to images whose filename starts with **E02** "
                            "(first 33 images only).  \n"
                            "_Conf / Val-IoU / GT-IoU sliders on the left are shared._")

                        compare_state = gr.State([None, None, None])

                        # ── Model path inputs ─────────────────────────────────
                        with gr.Row():
                            with gr.Column():
                                cmp_m1 = gr.Textbox(
                                    label="Model 1 (.pt)",
                                    placeholder=r"D:\models\best_v1.pt",
                                    interactive=True)
                                btn_browse_cmp1 = gr.Button("Browse", size="sm")
                            with gr.Column():
                                cmp_m2 = gr.Textbox(
                                    label="Model 2 (.pt)",
                                    placeholder=r"D:\models\best_v2.pt",
                                    interactive=True)
                                btn_browse_cmp2 = gr.Button("Browse", size="sm")
                            with gr.Column():
                                cmp_m3 = gr.Textbox(
                                    label="Model 3 (.pt) — optional",
                                    placeholder=r"D:\models\best_v3.pt",
                                    interactive=True)
                                btn_browse_cmp3 = gr.Button("Browse", size="sm")

                        with gr.Row():
                            cmp_dataset = gr.Textbox(
                                label="Dataset folder",
                                placeholder=r"C:\datasets\cctv_test",
                                scale=4, interactive=True)
                            btn_browse_cmp_ds = gr.Button("Browse", scale=1, min_width=60)

                        btn_compare_run = gr.Button(
                            "Run Comparison", variant="primary", size="lg")
                        compare_status = gr.Textbox(
                            label="Status", interactive=False, lines=2)

                        # ── Metrics cards ─────────────────────────────────────
                        gr.Markdown("### Overall Metrics")
                        with gr.Row():
                            cmp_md1 = gr.Markdown("_Model 1 not run_")
                            cmp_md2 = gr.Markdown("_Model 2 not run_")
                            cmp_md3 = gr.Markdown("_Model 3 not run_")

                        # ── Per-image comparison table ────────────────────────
                        gr.Markdown(
                            "### Falldown Yakalama Oranı — E02 subset\n"
                            "**GT_Fall** = o resimde kaç gerçek falldown var &nbsp;|&nbsp; "
                            "**Model sütunları** = kaç tanesini yakaladı &nbsp;|&nbsp; "
                            "En altta toplam ve % _(GT etiket dosyası gerekir)_")
                        cmp_table_out = gr.DataFrame(interactive=False, wrap=True)

                        # ── Image viewer ──────────────────────────────────────
                        gr.Markdown("### Image Viewer — select an E02 image")
                        cmp_img_dd = gr.Dropdown(
                            label="Image", choices=[], interactive=True)
                        with gr.Row():
                            cmp_img1 = gr.Image(
                                label="Model 1", type="pil",
                                height=480, show_download_button=True)
                            cmp_img2 = gr.Image(
                                label="Model 2", type="pil",
                                height=480, show_download_button=True)
                            cmp_img3 = gr.Image(
                                label="Model 3", type="pil",
                                height=480, show_download_button=True)

        # ── Wire-up ───────────────────────────────────────────────────────────

        btn_browse_model.click(browse_model_cb,   inputs=[], outputs=[model_path_inp])
        btn_browse_dataset.click(browse_dataset_cb, inputs=[], outputs=[dataset_inp])

        # Upload: fill path inputs automatically when file is selected
        model_upload_file.change(model_upload_cb,
                                 inputs=[model_upload_file], outputs=[model_path_inp])
        dataset_upload_file.change(dataset_zip_upload_cb,
                                   inputs=[dataset_upload_file], outputs=[dataset_inp])

        btn_load.click(load_model_cb, inputs=[model_path_inp],
                       outputs=[model_status, classes_box])

        # Validation
        btn_val.click(
            lambda: "Running validation...", inputs=[], outputs=[run_status]
        ).then(
            run_val_cb, inputs=[dataset_inp, conf_slider, iou_slider],
            outputs=[run_status, val_state, val_class_df],
        ).then(_history_df, inputs=[], outputs=[history_df])

        # Per-Image predict
        btn_predict.click(
            lambda: "Running analysis...", inputs=[], outputs=[run_status]
        ).then(
            run_predict_cb, inputs=[dataset_inp, conf_slider, iou_thresh_slider],
            outputs=[run_status, predict_state, has_gt_state,
                     per_image_df, img_dropdown, worst_df_out],
        ).then(_history_df, inputs=[], outputs=[history_df])

        # Run Both
        btn_both.click(
            lambda: "Running validation + prediction...", inputs=[], outputs=[run_status]
        ).then(
            run_both_cb,
            inputs=[dataset_inp, conf_slider, iou_slider, iou_thresh_slider],
            outputs=[run_status, val_state, val_class_df,
                     predict_state, has_gt_state, per_image_df, img_dropdown, worst_df_out],
        ).then(_history_df, inputs=[], outputs=[history_df])

        # Image viewer — new image selected: reset filter, load all detections
        view_inputs_load   = [img_dropdown, predict_state]
        view_outputs_load  = [annotated_img, det_df, class_filter_dd]

        img_dropdown.change(image_load_cb, view_inputs_load, view_outputs_load)
        btn_view.click(image_load_cb, view_inputs_load, view_outputs_load)

        # Class filter changed: re-draw with filter (keep selection)
        class_filter_dd.change(
            image_filter_cb,
            inputs=[img_dropdown, predict_state, class_filter_dd],
            outputs=[annotated_img, det_df],
        )

        # Row click in detection table → highlight that detection
        det_df.select(
            on_det_select,
            inputs=[img_dropdown, predict_state, class_filter_dd],
            outputs=[annotated_img],
        )

        # GT Comparison
        btn_compare.click(view_comparison_cb,
                          [img_dropdown, predict_state, has_gt_state],
                          [comparison_img, compare_df])

        # History
        btn_refresh_history.click(_history_df, inputs=[], outputs=[history_df])
        btn_clear_history.click(clear_history_cb, inputs=[], outputs=[history_df])
        btn_export_history.click(
            export_history_csv_cb, inputs=[],
            outputs=[history_export_status, history_export_file],
        ).then(
            lambda s, f: (gr.update(visible=bool(s), value=s),
                          gr.update(visible=f is not None, value=f)),
            inputs=[history_export_status, history_export_file],
            outputs=[history_export_status, history_export_file],
        )

        # Export
        btn_export.click(export_cb, inputs=[], outputs=[export_status, export_file])

        # Model Comparison — browse buttons
        btn_browse_cmp1.click(browse_model_cb, inputs=[], outputs=[cmp_m1])
        btn_browse_cmp2.click(browse_model_cb, inputs=[], outputs=[cmp_m2])
        btn_browse_cmp3.click(browse_model_cb, inputs=[], outputs=[cmp_m3])
        btn_browse_cmp_ds.click(browse_dataset_cb, inputs=[], outputs=[cmp_dataset])

        # Model Comparison — run
        btn_compare_run.click(
            lambda: "Running comparison — this may take several minutes...",
            inputs=[], outputs=[compare_status],
        ).then(
            run_compare_cb,
            inputs=[cmp_m1, cmp_m2, cmp_m3, cmp_dataset,
                    conf_slider, iou_slider, iou_thresh_slider],
            outputs=[compare_status, compare_state,
                     cmp_md1, cmp_md2, cmp_md3,
                     cmp_table_out, cmp_img_dd],
        )

        # Model Comparison — image viewer
        cmp_img_dd.change(
            compare_image_cb,
            inputs=[cmp_img_dd, compare_state],
            outputs=[cmp_img1, cmp_img2, cmp_img3],
        )

        # Load history on app start
        demo.load(_history_df, inputs=[], outputs=[history_df])

    return demo


if __name__ == "__main__":
    print("=" * 55)
    print(f"  Frontend  →  http://localhost:7861")
    print(f"  Backend   →  {API_BASE}")
    print("=" * 55)
    build_demo().launch(server_name="0.0.0.0", server_port=7861)
