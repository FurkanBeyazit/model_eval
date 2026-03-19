"""
main.py  –  Model Eval Platform — FastAPI Backend
Run: python backend/main.py
API docs: http://localhost:8000/docs
"""

import sys
from pathlib import Path

# Ensure backend/ is on the path so routers can import state / evaluator / exporter
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routers import model, analysis, export, upload

app = FastAPI(
    title="Model Eval Platform API",
    description="YOLO model evaluation REST API. Frontend-agnostic.",
    version="0.1.0",
)

# Allow any origin so Gradio (or Svelte/React on a different port) can call freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model.router)
app.include_router(analysis.router)
app.include_router(export.router)
app.include_router(upload.router)


@app.get("/health", tags=["health"])
def health():
    import state
    return {
        "status": "ok",
        "model_loaded": state.evaluator.model is not None,
        "model_path": state.evaluator.model_path,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  Backend  →  http://localhost:8001")
    print("  API docs →  http://localhost:8001/docs")
    print("=" * 55)
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
