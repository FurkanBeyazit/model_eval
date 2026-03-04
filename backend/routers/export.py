from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import state
from exporter import export_to_excel

router = APIRouter(prefix="/api/export", tags=["export"])


@router.post("/excel")
def export_excel():
    if not state.last_predict_results and not state.last_val_metrics:
        raise HTTPException(400, "No results to export. Run analysis first.")
    output_path = export_to_excel(
        state.last_predict_results,
        state.last_val_metrics,
        state.evaluator.class_names,
    )
    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="model_eval_results.xlsx",
    )
