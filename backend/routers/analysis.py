import io
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import state

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


# ── Request models ────────────────────────────────────────────────────────────

class ValidateRequest(BaseModel):
    dataset_path: str
    conf: float = 0.25
    iou: float = 0.45


class PredictRequest(BaseModel):
    dataset_path: str
    conf: float = 0.25
    iou_thresh: float = 0.5


class BothRequest(BaseModel):
    dataset_path: str
    conf: float = 0.25
    iou: float = 0.45
    iou_thresh: float = 0.5


# ── History helpers ───────────────────────────────────────────────────────────

def _make_predict_entry(dataset_path: str, conf: float, iou_thresh: float,
                        results: list, has_gt: bool, run_type: str = "predict",
                        val_metrics: dict = None) -> dict:
    """Build a single history row from a completed run."""
    total_pred = sum(r["total_detections"] for r in results)
    total_gt   = sum(sum(r.get("gt_counts", {}).values()) for r in results)
    total_tp   = sum(r.get("tp", 0) for r in results)
    total_fp   = sum(r.get("fp", 0) for r in results)
    total_fn   = sum(r.get("fn", 0) for r in results)
    precision  = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else ""
    recall     = round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else ""
    f1         = round(2 * precision * recall / (precision + recall), 4) \
                 if isinstance(precision, float) and isinstance(recall, float) \
                 and (precision + recall) > 0 else ""

    entry: dict = {
        "Timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Run_Type":    run_type,
        "Dataset":     dataset_path,
        "Conf":        conf,
        "GT_IoU":      iou_thresh if has_gt else "",
        "Images":      len(results),
        "Detections":  total_pred,
        "Has_GT":      has_gt,
    }
    if has_gt:
        entry.update({
            "GT_Objects": total_gt,
            "TP":         total_tp,
            "FP":         total_fp,
            "FN":         total_fn,
            "Precision":  precision,
            "Recall":     recall,
            "F1":         f1,
        })
    if val_metrics:
        entry["mAP50"]    = val_metrics.get("map50", "")
        entry["mAP50-95"] = val_metrics.get("map50_95", "")
        entry["Val_IoU"]  = val_metrics.get("iou", "")
    return entry


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/validate")
def validate(req: ValidateRequest):
    if state.evaluator.model is None:
        raise HTTPException(400, "Model not loaded. Call /api/model/load first.")
    metrics = state.evaluator.run_validation(req.dataset_path, req.conf, req.iou)
    state.last_val_metrics = metrics

    # Log to history
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Run_Type":  "validation",
        "Dataset":   req.dataset_path,
        "Conf":      req.conf,
        "Val_IoU":   req.iou,
        "mAP50":     metrics.get("map50", ""),
        "mAP50-95":  metrics.get("map50_95", ""),
        "Precision":  metrics.get("mp", ""),
        "Recall":    metrics.get("mr", ""),
    }
    state.run_history.append(entry)
    state.save_history(state.run_history)
    return metrics


@router.post("/predict")
def predict(req: PredictRequest):
    if state.evaluator.model is None:
        raise HTTPException(400, "Model not loaded. Call /api/model/load first.")
    results, raw_data, has_gt = state.evaluator.run_predict_with_gt(
        req.dataset_path, req.conf, req.iou_thresh
    )
    state.last_predict_results   = results
    state.last_raw_data          = raw_data
    state.has_gt                 = has_gt
    state.last_threshold_results = []

    # Log to history
    entry = _make_predict_entry(req.dataset_path, req.conf, req.iou_thresh,
                                results, has_gt, run_type="predict")
    state.run_history.append(entry)
    state.save_history(state.run_history)

    return {
        "total_images":     len(results),
        "total_detections": sum(r["total_detections"] for r in results),
        "has_gt":           has_gt,
        "results":          results,
    }


@router.post("/both")
def both(req: BothRequest):
    if state.evaluator.model is None:
        raise HTTPException(400, "Model not loaded. Call /api/model/load first.")
    val_metrics = state.evaluator.run_validation(req.dataset_path, req.conf, req.iou)
    state.last_val_metrics = val_metrics

    results, raw_data, has_gt = state.evaluator.run_predict_with_gt(
        req.dataset_path, req.conf, req.iou_thresh
    )
    state.last_predict_results   = results
    state.last_raw_data          = raw_data
    state.has_gt                 = has_gt
    state.last_threshold_results = []

    # Log to history
    entry = _make_predict_entry(req.dataset_path, req.conf, req.iou_thresh,
                                results, has_gt, run_type="both",
                                val_metrics={**val_metrics, "iou": req.iou})
    state.run_history.append(entry)
    state.save_history(state.run_history)

    return {
        "val_metrics": val_metrics,
        "predict": {
            "total_images":     len(results),
            "total_detections": sum(r["total_detections"] for r in results),
            "has_gt":           has_gt,
            "results":          results,
        },
    }


@router.get("/image/list")
def list_images():
    return {"images": [r["image_name"] for r in state.last_predict_results]}


@router.get("/image/annotated")
def get_annotated_image(image_name: str):
    result = next(
        (r for r in state.last_predict_results if r["image_name"] == image_name), None
    )
    if result is None:
        raise HTTPException(404, f"'{image_name}' not found in last predict results.")
    pil_img = state.evaluator.get_annotated_image(result["image_path"], result["detections"])
    if pil_img is None:
        raise HTTPException(500, f"Could not read image file: {result['image_path']}")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@router.get("/image/comparison")
def get_comparison_image(image_name: str):
    if not state.has_gt:
        raise HTTPException(400, "No GT labels found for this dataset.")
    result = next(
        (r for r in state.last_predict_results if r["image_name"] == image_name), None
    )
    if result is None:
        raise HTTPException(404, f"'{image_name}' not found in last predict results.")
    pil_img = state.evaluator.get_comparison_image(
        result["image_path"], result["tp_pairs"], result["fp_preds"], result["fn_gts"]
    )
    if pil_img is None:
        raise HTTPException(500, f"Could not read image file: {result['image_path']}")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


# ── History endpoints ─────────────────────────────────────────────────────────

@router.get("/history")
def get_history():
    return {"history": state.run_history}


@router.delete("/history")
def clear_history():
    state.run_history.clear()
    state.save_history(state.run_history)
    return {"ok": True}
