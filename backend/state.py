"""
state.py  –  Shared singleton state for the backend process.
For single-user local use. Replace with Redis / DB for multi-user.
"""

from evaluator import ModelEvaluator
from typing import List, Optional

evaluator              = ModelEvaluator()
last_predict_results:   List[dict]     = []
last_val_metrics:       Optional[dict] = None
last_threshold_results: List[dict]     = []   # populated lazily during export
last_raw_data:          List[dict]     = []   # per-image preds+GT at conf=0.01
has_gt:                 bool           = False
