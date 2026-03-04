import io
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/validate")
def validate(req: ValidateRequest):
    if state.evaluator.model is None:
        raise HTTPException(400, "Model not loaded. Call /api/model/load first.")
    metrics = state.evaluator.run_validation(req.dataset_path, req.conf, req.iou)
    state.last_val_metrics = metrics
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
    state.last_threshold_results = []   # invalidate cache
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
    state.last_threshold_results = []   # invalidate cache

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
    """Return annotated image (JPEG) for a given image name from the last predict run."""
    result = next(
        (r for r in state.last_predict_results if r["image_name"] == image_name),
        None,
    )
    if result is None:
        raise HTTPException(404, f"'{image_name}' not found in last predict results.")

    pil_img = state.evaluator.get_annotated_image(
        result["image_path"], result["detections"]
    )
    if pil_img is None:
        raise HTTPException(500, f"Could not read image file: {result['image_path']}")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@router.get("/image/comparison")
def get_comparison_image(image_name: str):
    """
    Return GT-vs-prediction comparison image (JPEG).
    Green = matched GT, Lime dashed = TP pred, Red = FP, Orange = FN (missed GT).
    Only available when GT labels exist.
    """
    if not state.has_gt:
        raise HTTPException(400, "No GT labels found for this dataset.")

    result = next(
        (r for r in state.last_predict_results if r["image_name"] == image_name),
        None,
    )
    if result is None:
        raise HTTPException(404, f"'{image_name}' not found in last predict results.")

    pil_img = state.evaluator.get_comparison_image(
        result["image_path"],
        result["tp_pairs"],
        result["fp_preds"],
        result["fn_gts"],
    )
    if pil_img is None:
        raise HTTPException(500, f"Could not read image file: {result['image_path']}")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
