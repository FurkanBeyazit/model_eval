from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import state
from exporter import export_to_excel

router = APIRouter(prefix="/api/export", tags=["export"])


@router.post("/excel")
def export_excel():
    if not state.last_predict_results and not state.last_val_metrics:
        raise HTTPException(400, "No results to export. Run analysis first.")

    # ── Threshold curve: run lazily on first export ───────────────────────────
    # Uses raw_data (preds at conf=0.01 + GT boxes) for real P/R/F1.
    # Falls back to directory-based predict if no raw_data cached.
    if state.last_predict_results and not state.last_threshold_results:
        try:
            if state.last_raw_data:
                state.last_threshold_results = state.evaluator.run_threshold_analysis(
                    state.last_raw_data
                )
            else:
                img_dir = str(Path(state.last_predict_results[0]["image_path"]).parent)
                state.last_threshold_results = state.evaluator.run_threshold_analysis_from_dir(
                    img_dir
                )
        except Exception:
            state.last_threshold_results = []

    output_path = export_to_excel(
        state.last_predict_results,
        state.last_val_metrics,
        state.evaluator.class_names,
        threshold_results=state.last_threshold_results,
        has_gt=state.has_gt,
    )
    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="model_eval_results.xlsx",
    )
