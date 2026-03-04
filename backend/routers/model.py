from fastapi import APIRouter
from pydantic import BaseModel
import state

router = APIRouter(prefix="/api/model", tags=["model"])


class LoadRequest(BaseModel):
    model_path: str


@router.get("/status")
def get_status():
    return {
        "loaded": state.evaluator.model is not None,
        "model_path": state.evaluator.model_path,
        "num_classes": len(state.evaluator.class_names),
        "classes": {str(k): v for k, v in state.evaluator.class_names.items()},
    }


@router.post("/load")
def load_model(req: LoadRequest):
    ok, msg = state.evaluator.load_model(req.model_path)
    return {
        "ok": ok,
        "message": msg,
        "num_classes": len(state.evaluator.class_names),
        "classes": {str(k): v for k, v in state.evaluator.class_names.items()},
    }
