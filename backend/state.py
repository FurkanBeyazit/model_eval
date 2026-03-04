"""
state.py  –  Shared singleton state for the backend process.
For single-user local use. Replace with Redis / DB for multi-user.
"""

from evaluator import ModelEvaluator
from typing import List, Optional

evaluator = ModelEvaluator()
last_predict_results: List[dict] = []
last_val_metrics: Optional[dict] = None
