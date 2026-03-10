"""
state.py  –  Shared singleton state for the backend process.
For single-user local use. Replace with Redis / DB for multi-user.
"""

import json
from pathlib import Path
from evaluator import ModelEvaluator
from typing import List, Optional

# History persisted to disk between sessions
HISTORY_FILE = Path.home() / ".model_eval_history.json"


def _load_history() -> List[dict]:
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def save_history(history: List[dict]) -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


evaluator              = ModelEvaluator()
last_predict_results:   List[dict]     = []
last_val_metrics:       Optional[dict] = None
last_threshold_results: List[dict]     = []   # populated lazily during export
last_raw_data:          List[dict]     = []   # per-image preds+GT at conf=0.01
has_gt:                 bool           = False
run_history:            List[dict]     = _load_history()  # persisted run log
