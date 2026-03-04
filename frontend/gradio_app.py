"""
gradio_app.py  –  Model Eval Platform — Gradio Frontend
Communicates exclusively with the FastAPI backend via REST.
Swap this file for a Svelte/React app without touching backend/.

Run: python frontend/gradio_app.py
UI:  http://localhost:7860
"""

import os
import io
import tempfile
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import gradio as gr
import pandas as pd
import requests
from PIL import Image

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
TIMEOUT  = 600  # seconds — long for large datasets


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


# ── Native browse dialogs (tkinter — lokal makine) ───────────────────────────

def browse_model_cb():
    """Windows dosya seçici — .pt dosyası için."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askopenfilename(
        title="Model Seç (.pt)",
        filetypes=[("YOLO Model", "*.pt"), ("Tüm dosyalar", "*.*")],
    )
    root.destroy()
    return path or ""


def browse_dataset_cb():
    """Windows klasör seçici — dataset dizini için."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askdirectory(title="Dataset Klasörü Seç")
    root.destroy()
    return path or ""


# ── DataFrame builders ────────────────────────────────────────────────────────

def _per_image_df(results: list) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    all_classes = sorted({
        det["class_name"]
        for r in results
        for det in r["detections"]
    })
    rows = []
    for r in results:
        row = {"Image": r["image_name"], "Total": r["total_detections"]}
        for cls in all_classes:
            row[cls] = r["class_counts"].get(cls, 0)
            conf = r["class_avg_conf"].get(cls)
            row[f"{cls}_conf"] = f"{conf:.3f}" if conf is not None else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _stats_df(results: list) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    all_classes = sorted({
        det["class_name"]
        for r in results
        for det in r["detections"]
    })
    rows = []
    for cls in all_classes:
        confs = [
            det["confidence"]
            for r in results
            for det in r["detections"]
            if det["class_name"] == cls
        ]
        imgs_with = sum(1 for r in results if cls in r["class_counts"])
        rows.append({
            "Class":          cls,
            "Total Det.":     len(confs),
            "Images w/ Det.": imgs_with,
            "Avg Conf":       round(sum(confs) / len(confs), 3) if confs else 0,
            "Min Conf":       round(min(confs), 3) if confs else 0,
            "Max Conf":       round(max(confs), 3) if confs else 0,
        })
    return pd.DataFrame(rows)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def load_model_cb(model_path: str):
    if not model_path or not model_path.strip():
        return "Model path boş.", ""
    try:
        data = _post("/api/model/load", {"model_path": model_path})
        if data["ok"]:
            classes_lines = "\n".join(
                f"  [{k}] {v}"
                for k, v in sorted(data["classes"].items(), key=lambda x: int(x[0]))
            )
            return f"✅ {data['message']}", classes_lines
        return f"❌ {data['message']}", ""
    except Exception as exc:
        return f"❌ Backend hatası: {exc}", ""


def run_val_cb(dataset_path: str, conf: float, iou: float, progress=gr.Progress()):
    if not _check_backend():
        return "❌ Backend çalışmıyor (backend/main.py başlat)", {}, None, None
    try:
        progress(0.05, desc="Validation başlatılıyor…")
        data = _post("/api/analysis/validate", {"dataset_path": dataset_path, "conf": conf, "iou": iou})
        progress(1.0, desc="Tamamlandı!")
        df_summary = pd.DataFrame([
            {"Metric": "mAP50",          "Value": data["map50"]},
            {"Metric": "mAP50-95",       "Value": data["map50_95"]},
            {"Metric": "Mean Precision",  "Value": data["mp"]},
            {"Metric": "Mean Recall",     "Value": data["mr"]},
        ])
        df_class = pd.DataFrame(data.get("class_metrics", []))
        return "✅ Validation tamamlandı!", data, df_summary, df_class
    except Exception as exc:
        return f"❌ {exc}", {}, None, None


def run_predict_cb(dataset_path: str, conf: float, progress=gr.Progress()):
    if not _check_backend():
        return "❌ Backend çalışmıyor", [], None, gr.update(choices=[], value=None)
    try:
        progress(0.05, desc="Görseller taranıyor…")
        data = _post("/api/analysis/predict", {"dataset_path": dataset_path, "conf": conf})
        progress(0.90, desc="Tablo oluşturuluyor…")
        results = data["results"]
        df = _per_image_df(results)
        names = [r["image_name"] for r in results]
        progress(1.0, desc="Tamamlandı!")
        return (
            f"✅ {data['total_images']} görsel | {data['total_detections']} tespit",
            results,
            df,
            gr.update(choices=names, value=names[0] if names else None),
        )
    except Exception as exc:
        return f"❌ {exc}", [], None, gr.update(choices=[], value=None)


def run_both_cb(dataset_path: str, conf: float, iou: float, progress=gr.Progress()):
    if not _check_backend():
        empty = gr.update(choices=[], value=None)
        return "❌ Backend çalışmıyor", {}, None, None, [], None, empty
    try:
        progress(0.05, desc="Validation + prediction çalışıyor…")
        data = _post("/api/analysis/both", {"dataset_path": dataset_path, "conf": conf, "iou": iou})
        progress(0.90, desc="Tablolar oluşturuluyor…")
        val     = data["val_metrics"]
        pred    = data["predict"]
        results = pred["results"]
        df_summary = pd.DataFrame([
            {"Metric": "mAP50",          "Value": val["map50"]},
            {"Metric": "mAP50-95",       "Value": val["map50_95"]},
            {"Metric": "Mean Precision",  "Value": val["mp"]},
            {"Metric": "Mean Recall",     "Value": val["mr"]},
        ])
        df_class   = pd.DataFrame(val.get("class_metrics", []))
        df_per_img = _per_image_df(results)
        names = [r["image_name"] for r in results]
        progress(1.0, desc="Tamamlandı!")
        status = (
            f"✅ Val mAP50={val['map50']} | "
            f"Predict: {pred['total_images']} görsel, {pred['total_detections']} tespit"
        )
        return (
            status,
            val, df_summary, df_class,
            results, df_per_img,
            gr.update(choices=names, value=names[0] if names else None),
        )
    except Exception as exc:
        empty = gr.update(choices=[], value=None)
        return f"❌ {exc}", {}, None, None, [], None, empty


def view_image_cb(image_name: str, predict_state: list):
    if not image_name or not predict_state:
        return None, pd.DataFrame()
    pil_img = None
    try:
        resp = _get("/api/analysis/image/annotated", {"image_name": image_name})
        pil_img = Image.open(io.BytesIO(resp.content))
    except Exception:
        pass
    result = next((r for r in predict_state if r["image_name"] == image_name), None)
    if result is None or not result["detections"]:
        return pil_img, pd.DataFrame(columns=["Class", "Conf", "X1", "Y1", "X2", "Y2", "Width", "Height"])
    df = pd.DataFrame([
        {
            "Class":  d["class_name"],
            "Conf":   d["confidence"],
            "X1": d["x1"], "Y1": d["y1"],
            "X2": d["x2"], "Y2": d["y2"],
            "Width":  round(d["x2"] - d["x1"], 1),
            "Height": round(d["y2"] - d["y1"], 1),
        }
        for d in result["detections"]
    ])
    return pil_img, df


def export_cb(progress=gr.Progress()):
    try:
        progress(0.2, desc="Excel oluşturuluyor…")
        resp = requests.post(f"{API_BASE}/api/export/excel", timeout=60)
        if resp.status_code != 200:
            return f"❌ {resp.json().get('detail', 'Hata')}", None
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.write(resp.content)
        tmp.close()
        progress(1.0, desc="Hazır!")
        return "✅ İndirilmeye hazır", tmp.name
    except Exception as exc:
        return f"❌ {exc}", None


# ── Gradio UI ─────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")
CSS = "footer { display: none !important; }"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Model Eval Platform", css=CSS, theme=THEME) as demo:

        gr.Markdown(
            "# 🔍 Model Eval Platform\n"
            "YOLO validation (aggregate metrics) + per-image analysis"
        )

        val_state     = gr.State({})
        predict_state = gr.State([])

        with gr.Row(equal_height=False):

            # ────────── LEFT: Config ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=300):

                gr.Markdown("### ⚙️ Model")
                with gr.Row():
                    model_path_inp = gr.Textbox(
                        label="Model path (.pt)",
                        placeholder=r"D:\models\best.pt",
                        lines=1,
                        scale=4,
                        interactive=True,
                    )
                    btn_browse_model = gr.Button("📁", scale=1, min_width=48)
                btn_load     = gr.Button("Model Yükle", variant="primary")
                model_status = gr.Textbox(label="Durum", interactive=False, lines=2)
                classes_box  = gr.Textbox(
                    label="Sınıflar", interactive=False, lines=6, max_lines=15
                )

                gr.Markdown("### 📂 Dataset")
                with gr.Row():
                    dataset_inp = gr.Textbox(
                        label="Dataset klasörü",
                        placeholder=r"C:\datasets\cctv_test",
                        info="Desteklenen yapı: root/val/images · root/images · root/",
                        lines=1,
                        scale=4,
                        interactive=True,
                    )
                    btn_browse_dataset = gr.Button("📁", scale=1, min_width=48)
                conf_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
                iou_slider  = gr.Slider(0.10, 0.95, value=0.45, step=0.05, label="IoU (val only)")

                gr.Markdown("### ▶️ Çalıştır")
                with gr.Row():
                    btn_val     = gr.Button("Validation",  variant="secondary", size="sm")
                    btn_predict = gr.Button("Per-Image",   variant="secondary", size="sm")
                btn_both    = gr.Button("İkisini Çalıştır ▶", variant="primary")
                run_status  = gr.Textbox(label="Durum", interactive=False, lines=2)

            # ────────── RIGHT: Results ────────────────────────────────────────
            with gr.Column(scale=3):
                with gr.Tabs():

                    with gr.Tab("📊 Validation Metrikleri"):
                        gr.Markdown("Aggregate metrikler `model.val()` — ground-truth etiket gerektirir.")
                        gr.Markdown("#### Genel")
                        val_summary_df = gr.DataFrame(interactive=False, wrap=True)
                        gr.Markdown("#### Sınıf Bazlı (P / R / F1 / mAP50 / mAP50-95)")
                        val_class_df   = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("🖼️ Görsel Bazlı Sonuçlar"):
                        gr.Markdown(
                            "Her görsel için tespit sayısı + ortalama confidence.  \n"
                            "`<sınıf>` = adet · `<sınıf>_conf` = ort. confidence"
                        )
                        per_image_df = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("🔎 Görsel İnceleme"):
                        gr.Markdown("Backend'den annotated görsel — Per-Image çalıştırdıktan sonra aktif.")
                        with gr.Row():
                            img_dropdown = gr.Dropdown(
                                label="Görsel seç", choices=[], interactive=True, scale=4
                            )
                            btn_view = gr.Button("Göster →", variant="secondary", scale=1)
                        with gr.Row():
                            annotated_img = gr.Image(label="Annotated", type="pil", scale=2)
                            det_df = gr.DataFrame(
                                label="Tespitler", interactive=False, scale=1, wrap=True
                            )

                    with gr.Tab("📈 Hızlı İstatistik"):
                        gr.Markdown("Sınıf bazlı confidence istatistikleri (son predict'ten).")
                        stats_df = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("💾 Dışa Aktar"):
                        gr.Markdown(
                            "**Excel çıktı sayfaları:**\n"
                            "- `Summary` — genel metrikler\n"
                            "- `Val_Class_Metrics` — P / R / F1 / mAP sınıf bazlı\n"
                            "- `Per_Image` — görsel başına count + confidence\n"
                            "- `All_Detections` — tüm bbox'lar (sınıf, conf, koordinat, boyut)\n"
                            "- `Class_Distribution` — sınıf başına confidence istatistikleri"
                        )
                        btn_export    = gr.Button("Excel Oluştur", variant="primary", size="lg")
                        export_status = gr.Textbox(label="Durum", interactive=False, lines=1)
                        export_file   = gr.File(label="İndir", interactive=False)

        # ── Wire-up ───────────────────────────────────────────────────────────

        btn_browse_model.click(browse_model_cb,   inputs=[], outputs=[model_path_inp])
        btn_browse_dataset.click(browse_dataset_cb, inputs=[], outputs=[dataset_inp])

        btn_load.click(
            load_model_cb,
            inputs=[model_path_inp],
            outputs=[model_status, classes_box],
        )

        btn_val.click(
            run_val_cb,
            inputs=[dataset_inp, conf_slider, iou_slider],
            outputs=[run_status, val_state, val_summary_df, val_class_df],
        )

        btn_predict.click(
            run_predict_cb,
            inputs=[dataset_inp, conf_slider],
            outputs=[run_status, predict_state, per_image_df, img_dropdown],
        ).then(
            _stats_df, inputs=[predict_state], outputs=[stats_df]
        )

        btn_both.click(
            run_both_cb,
            inputs=[dataset_inp, conf_slider, iou_slider],
            outputs=[
                run_status,
                val_state, val_summary_df, val_class_df,
                predict_state, per_image_df, img_dropdown,
            ],
        ).then(
            _stats_df, inputs=[predict_state], outputs=[stats_df]
        )

        view_inputs  = [img_dropdown, predict_state]
        view_outputs = [annotated_img, det_df]
        img_dropdown.change(view_image_cb, view_inputs, view_outputs)
        btn_view.click(view_image_cb, view_inputs, view_outputs)

        btn_export.click(
            export_cb,
            inputs=[],
            outputs=[export_status, export_file],
        )

    return demo


if __name__ == "__main__":
    print("=" * 55)
    print(f"  Frontend  →  http://localhost:7860")
    print(f"  Backend   →  {API_BASE}")
    print("=" * 55)
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
