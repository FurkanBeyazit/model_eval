"""
gradio_app.py  –  Model Eval Platform — Gradio Frontend
Communicates exclusively with the FastAPI backend via REST.
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

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
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


def _stats_df(results: list) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    all_classes = sorted({det["class_name"] for r in results for det in r["detections"]})
    rows = []
    for cls in all_classes:
        confs = [det["confidence"] for r in results for det in r["detections"]
                 if det["class_name"] == cls]
        rows.append({
            "Class":          cls,
            "Total Det.":     len(confs),
            "Images w/ Det.": sum(1 for r in results if cls in r["class_counts"]),
            "Avg Conf":       round(sum(confs) / len(confs), 3) if confs else 0,
            "Min Conf":       round(min(confs), 3) if confs else 0,
            "Max Conf":       round(max(confs), 3) if confs else 0,
        })
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


def _history_df() -> pd.DataFrame:
    try:
        data = _get("/api/analysis/history").json()
        rows = data.get("history", [])
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── Callbacks ─────────────────────────────────────────────────────────────────

def load_model_cb(model_path: str):
    if not model_path or not model_path.strip():
        return "Model path is empty.", ""
    try:
        data = _post("/api/model/load", {"model_path": model_path})
        if data["ok"]:
            classes_lines = "\n".join(
                f"  [{k}] {v}"
                for k, v in sorted(data["classes"].items(), key=lambda x: int(x[0]))
            )
            return f"Loaded: {data['message']}", classes_lines
        return f"Error: {data['message']}", ""
    except Exception as exc:
        return f"Backend error: {exc}", ""


def run_val_cb(dataset_path: str, conf: float, iou: float):
    if not _check_backend():
        return "Backend not running — start backend/main.py first.", {}, None, None
    try:
        data = _post("/api/analysis/validate",
                     {"dataset_path": dataset_path, "conf": conf, "iou": iou})
        df_summary = pd.DataFrame([
            {"Metric": "mAP50",         "Value": data["map50"]},
            {"Metric": "mAP50-95",      "Value": data["map50_95"]},
            {"Metric": "Mean Precision", "Value": data["mp"]},
            {"Metric": "Mean Recall",    "Value": data["mr"]},
        ])
        df_class = pd.DataFrame(data.get("class_metrics", []))
        return "Validation complete.", data, df_summary, df_class
    except Exception as exc:
        return f"Error: {exc}", {}, None, None


def run_predict_cb(dataset_path: str, conf: float, iou_thresh: float):
    if not _check_backend():
        return "Backend not running.", [], False, None, gr.update(choices=[], value=None), None
    try:
        data = _post("/api/analysis/predict",
                     {"dataset_path": dataset_path, "conf": conf, "iou_thresh": iou_thresh})
        results  = data["results"]
        has_gt   = data.get("has_gt", False)
        df       = _per_image_df(results, has_gt)
        names    = [r["image_name"] for r in results]
        worst_df = _worst_images_df(results, has_gt)
        gt_note  = " | GT labels found ✓" if has_gt else " | No GT labels"
        return (
            f"{data['total_images']} images | {data['total_detections']} detections{gt_note}",
            results, has_gt, df,
            gr.update(choices=names, value=names[0] if names else None),
            worst_df,
        )
    except Exception as exc:
        return f"Error: {exc}", [], False, None, gr.update(choices=[], value=None), None


def run_both_cb(dataset_path: str, conf: float, iou: float, iou_thresh: float):
    if not _check_backend():
        empty = gr.update(choices=[], value=None)
        return "Backend not running.", {}, None, None, [], False, None, empty, None
    try:
        data = _post("/api/analysis/both", {
            "dataset_path": dataset_path, "conf": conf,
            "iou": iou, "iou_thresh": iou_thresh,
        })
        val     = data["val_metrics"]
        pred    = data["predict"]
        results = pred["results"]
        has_gt  = pred.get("has_gt", False)
        df_summary = pd.DataFrame([
            {"Metric": "mAP50",         "Value": val["map50"]},
            {"Metric": "mAP50-95",      "Value": val["map50_95"]},
            {"Metric": "Mean Precision", "Value": val["mp"]},
            {"Metric": "Mean Recall",    "Value": val["mr"]},
        ])
        df_class   = pd.DataFrame(val.get("class_metrics", []))
        df_per_img = _per_image_df(results, has_gt)
        worst_df   = _worst_images_df(results, has_gt)
        names = [r["image_name"] for r in results]
        gt_note = " | GT labels found ✓" if has_gt else " | No GT labels"
        status = (f"Val mAP50={val['map50']} | "
                  f"Predict: {pred['total_images']} images, "
                  f"{pred['total_detections']} detections{gt_note}")
        return (
            status,
            val, df_summary, df_class,
            results, has_gt, df_per_img,
            gr.update(choices=names, value=names[0] if names else None),
            worst_df,
        )
    except Exception as exc:
        empty = gr.update(choices=[], value=None)
        return f"Error: {exc}", {}, None, None, [], False, None, empty, None


def view_image_cb(image_name: str, predict_state: list):
    if not image_name or not predict_state:
        return None, pd.DataFrame()
    pil_img = None
    try:
        resp    = _get("/api/analysis/image/annotated", {"image_name": image_name})
        pil_img = Image.open(io.BytesIO(resp.content))
    except Exception:
        pass
    result = next((r for r in predict_state if r["image_name"] == image_name), None)
    if result is None or not result["detections"]:
        return pil_img, pd.DataFrame(columns=["Class", "Conf", "X1", "Y1", "X2", "Y2"])
    df = pd.DataFrame([
        {"Class": d["class_name"], "Conf": d["confidence"],
         "X1": d["x1"], "Y1": d["y1"], "X2": d["x2"], "Y2": d["y2"]}
        for d in result["detections"]
    ])
    return pil_img, df


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
                     "Conf": pair["pred"]["confidence"], "IoU": pair["iou"]})
    for p in result.get("fp_preds", []):
        rows.append({"Type": "FP", "Class": p["class_name"], "Conf": p["confidence"], "IoU": ""})
    for g in result.get("fn_gts", []):
        rows.append({"Type": "FN (Miss)", "Class": g["class_name"], "Conf": "", "IoU": ""})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Type", "Class", "Conf", "IoU"])
    return pil_img, df


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


def clear_history_cb():
    try:
        requests.delete(f"{API_BASE}/api/analysis/history", timeout=10)
    except Exception:
        pass
    return pd.DataFrame()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")
CSS   = "footer { display: none !important; }"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Model Eval Platform", css=CSS, theme=THEME) as demo:

        gr.Markdown("# Model Eval Platform\nYOLO fine-tuned model evaluation")

        val_state     = gr.State({})
        predict_state = gr.State([])
        has_gt_state  = gr.State(False)

        with gr.Row(equal_height=False):

            # ── Left: Config ──────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=300):

                gr.Markdown("### Model")
                with gr.Row():
                    model_path_inp   = gr.Textbox(label="Model path (.pt)",
                                                  placeholder=r"D:\models\best.pt",
                                                  lines=1, scale=4, interactive=True)
                    btn_browse_model = gr.Button("Browse", scale=1, min_width=64)
                btn_load     = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Status", interactive=False, lines=2)
                classes_box  = gr.Textbox(label="Classes", interactive=False,
                                          lines=6, max_lines=15)

                gr.Markdown("### Dataset")
                with gr.Row():
                    dataset_inp        = gr.Textbox(
                        label="Dataset folder",
                        placeholder=r"C:\datasets\cctv_test",
                        info="Supported: root/val/images · root/images · root/",
                        lines=1, scale=4, interactive=True)
                    btn_browse_dataset = gr.Button("Browse", scale=1, min_width=64)

                conf_slider = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.05,
                    label="Confidence threshold",
                    info="Keep predictions with score ≥ this. "
                         "0.25 = accept only 25%+ confident detections.")

                iou_slider = gr.Slider(
                    0.10, 0.95, value=0.45, step=0.05,
                    label="IoU threshold  (Validation only)",
                    info="Used internally by model.val() for NMS. "
                         "Does NOT affect GT label matching.")

                iou_thresh_slider = gr.Slider(
                    0.10, 0.90, value=0.50, step=0.05,
                    label="GT match IoU  (TP / FP / FN)",
                    info="Spatial overlap needed to call a detection a True Positive. "
                         "Example: 0.5 means the predicted box must overlap its GT box "
                         "by ≥ 50% of the union area. Has nothing to do with confidence.")

                gr.Markdown("### Run")
                with gr.Row():
                    btn_val     = gr.Button("Validation",  variant="secondary", size="sm")
                    btn_predict = gr.Button("Per-Image",   variant="secondary", size="sm")
                btn_both   = gr.Button("Run Both", variant="primary")
                run_status = gr.Textbox(label="Status", interactive=False, lines=2)

            # ── Right: Results ────────────────────────────────────────────────
            with gr.Column(scale=3):
                with gr.Tabs():

                    with gr.Tab("Validation Metrics"):
                        gr.Markdown("Aggregate metrics from `model.val()` — requires GT label files.")
                        gr.Markdown("#### Overall")
                        val_summary_df = gr.DataFrame(interactive=False, wrap=True)
                        gr.Markdown("#### Per-Class  (P / R / F1 / mAP50 / mAP50-95)")
                        val_class_df   = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("Per-Image Results"):
                        gr.Markdown(
                            "Detection counts per image. "
                            "GT columns appear when label files are found.")
                        per_image_df = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("Image Viewer"):
                        gr.Markdown("Available after running Per-Image or Run Both.")
                        with gr.Row():
                            img_dropdown = gr.Dropdown(
                                label="Select image", choices=[], interactive=True, scale=4)
                            btn_view = gr.Button("Show Annotated", variant="secondary", scale=1)
                        with gr.Tabs():
                            with gr.Tab("Prediction (Annotated)"):
                                with gr.Row():
                                    annotated_img = gr.Image(label="Annotated",
                                                             type="pil", scale=2)
                                    det_df = gr.DataFrame(label="Detections",
                                                          interactive=False, scale=1, wrap=True)
                            with gr.Tab("GT Comparison"):
                                gr.Markdown(
                                    "**Colour key:**  Green = matched GT  |  "
                                    "Lime dashed = TP pred  |  Red = FP  |  Orange = FN (missed)\n\n"
                                    "_GT label (.txt) files must exist in the dataset._")
                                btn_compare = gr.Button("Show GT Comparison", variant="primary")
                                with gr.Row():
                                    comparison_img = gr.Image(label="GT vs Prediction",
                                                              type="pil", scale=2)
                                    compare_df = gr.DataFrame(label="TP / FP / FN Detail",
                                                              interactive=False, scale=1, wrap=True)

                    with gr.Tab("Worst Images"):
                        gr.Markdown(
                            "GT available → sorted by most missed (FN).  \n"
                            "No GT → zero-detection + lowest confidence.  _(Top 20)_")
                        worst_df_out = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("Quick Stats"):
                        gr.Markdown("Per-class confidence statistics from the last predict run.")
                        stats_df = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("Run History"):
                        gr.Markdown(
                            "Every predict / validation run is logged here.  \n"
                            "Persists across sessions (saved to `~/.model_eval_history.json`).")
                        with gr.Row():
                            btn_refresh_history = gr.Button("Refresh", variant="secondary",
                                                            size="sm")
                            btn_clear_history   = gr.Button("Clear History", variant="stop",
                                                            size="sm")
                        history_df = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("Export"):
                        gr.Markdown(
                            "**Excel sheets:** Summary · Val_Class_Metrics · Class_Performance · "
                            "Per_Image · Per_Image_GT · All_Detections · Class_Distribution · "
                            "Confusion_Matrix · Threshold_Curve · Size_Analysis · "
                            "Worst_Images · Spatial_Bias")
                        btn_export    = gr.Button("Generate Excel", variant="primary", size="lg")
                        export_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        export_file   = gr.File(label="Download", interactive=False)

        # ── Wire-up ───────────────────────────────────────────────────────────

        btn_browse_model.click(browse_model_cb,   inputs=[], outputs=[model_path_inp])
        btn_browse_dataset.click(browse_dataset_cb, inputs=[], outputs=[dataset_inp])
        btn_load.click(load_model_cb, inputs=[model_path_inp],
                       outputs=[model_status, classes_box])

        # Validation — "Running..." on click, result on completion, history refresh
        btn_val.click(
            lambda: "Running validation...", inputs=[], outputs=[run_status]
        ).then(
            run_val_cb,
            inputs=[dataset_inp, conf_slider, iou_slider],
            outputs=[run_status, val_state, val_summary_df, val_class_df],
        ).then(
            _history_df, inputs=[], outputs=[history_df]
        )

        # Per-Image predict
        btn_predict.click(
            lambda: "Running analysis...", inputs=[], outputs=[run_status]
        ).then(
            run_predict_cb,
            inputs=[dataset_inp, conf_slider, iou_thresh_slider],
            outputs=[run_status, predict_state, has_gt_state,
                     per_image_df, img_dropdown, worst_df_out],
        ).then(
            _stats_df, inputs=[predict_state], outputs=[stats_df]
        ).then(
            _history_df, inputs=[], outputs=[history_df]
        )

        # Run Both
        btn_both.click(
            lambda: "Running validation + prediction...", inputs=[], outputs=[run_status]
        ).then(
            run_both_cb,
            inputs=[dataset_inp, conf_slider, iou_slider, iou_thresh_slider],
            outputs=[run_status, val_state, val_summary_df, val_class_df,
                     predict_state, has_gt_state, per_image_df, img_dropdown, worst_df_out],
        ).then(
            _stats_df, inputs=[predict_state], outputs=[stats_df]
        ).then(
            _history_df, inputs=[], outputs=[history_df]
        )

        # Image viewer
        img_dropdown.change(view_image_cb, [img_dropdown, predict_state], [annotated_img, det_df])
        btn_view.click(view_image_cb, [img_dropdown, predict_state], [annotated_img, det_df])
        btn_compare.click(view_comparison_cb,
                          [img_dropdown, predict_state, has_gt_state],
                          [comparison_img, compare_df])

        # History tab
        btn_refresh_history.click(_history_df, inputs=[], outputs=[history_df])
        btn_clear_history.click(clear_history_cb, inputs=[], outputs=[history_df])

        # Export
        btn_export.click(export_cb, inputs=[], outputs=[export_status, export_file])

        # Load history on startup
        demo.load(_history_df, inputs=[], outputs=[history_df])

    return demo


if __name__ == "__main__":
    print("=" * 55)
    print(f"  Frontend  →  http://localhost:7860")
    print(f"  Backend   →  {API_BASE}")
    print("=" * 55)
    build_demo().launch(server_name="0.0.0.0", server_port=7860, share=True)
