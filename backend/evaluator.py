"""
evaluator.py  –  YOLO Model Evaluator
Handles model loading, validation (model.val), per-image prediction (model.predict)
and image annotation.
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import tempfile
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class ModelEvaluator:
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.model_path: Optional[str] = None
        self.class_names: Dict[int, str] = {}

    # ─────────────────────────────────── Model ───────────────────────────────────

    def load_model(self, model_path: str) -> Tuple[bool, str]:
        path_str = model_path.strip().strip('"').strip("'")
        path = Path(path_str)
        if not path.exists():
            return False, f"File not found: {path_str}"
        try:
            self.model = YOLO(str(path))
            self.model_path = str(path)
            self.class_names = self.model.names  # {0: 'person', ...}
            names_preview = ", ".join(
                f"{k}:{v}" for k, v in sorted(self.class_names.items())
            )
            return True, f"{len(self.class_names)} classes — {names_preview}"
        except Exception as exc:
            self.model = None
            return False, f"Error loading model: {exc}"

    # ────────────────────────────────── Helpers ──────────────────────────────────

    def _find_image_dir(self, dataset_path: str) -> Optional[Path]:
        """Return the first directory that contains images, searching common layouts."""
        root = Path(dataset_path.strip().strip('"').strip("'"))
        if not root.exists():
            return None

        candidates = [
            root / "val" / "images",
            root / "test" / "images",
            root / "images",
            root / "val",
            root / "test",
            root,
        ]
        for candidate in candidates:
            if candidate.exists():
                imgs = [f for f in candidate.iterdir() if f.suffix.lower() in IMAGE_EXTS]
                if imgs:
                    return candidate
        return root  # fallback – let ultralytics handle the error

    def _create_yaml(self, dataset_path: str) -> str:
        """Create a temporary YAML for model.val()."""
        root = Path(dataset_path.strip().strip('"').strip("'"))

        # Detect val images relative path
        for rel in ("val/images", "test/images", "images", "val", "."):
            if (root / rel).exists():
                val_rel = rel
                break
        else:
            val_rel = "."

        yaml_content = {
            "path": str(root.absolute()),
            "train": val_rel,
            "val": val_rel,
            "nc": len(self.class_names),
            "names": {int(k): v for k, v in self.class_names.items()},
        }
        yaml_path = Path(tempfile.gettempdir()) / "model_eval_temp.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True)
        return str(yaml_path)

    # ──────────────────────────────── Validation ─────────────────────────────────

    def run_validation(self, dataset_path: str, conf: float = 0.25, iou: float = 0.45) -> dict:
        """Run model.val() and return structured metrics dict."""
        yaml_file = self._create_yaml(dataset_path)

        metrics = self.model.val(
            data=yaml_file,
            conf=conf,
            iou=iou,
            split="val",
            verbose=True,
        )

        # Per-class breakdown — ultralytics stores these indexed by ap_class_index
        ap_idx = (
            metrics.box.ap_class_index.tolist()
            if hasattr(metrics.box.ap_class_index, "tolist")
            else list(metrics.box.ap_class_index)
        )

        class_metrics = []
        for i, cls_id in enumerate(ap_idx):
            cls_name = metrics.names.get(int(cls_id), f"class_{cls_id}")
            class_metrics.append(
                {
                    "class": cls_name,
                    "class_id": int(cls_id),
                    "Precision": round(float(metrics.box.p[i]), 4),
                    "Recall": round(float(metrics.box.r[i]), 4),
                    "F1": round(float(metrics.box.f1[i]), 4),
                    "mAP50": round(float(metrics.box.ap50[i]), 4),
                    "mAP50-95": round(float(metrics.box.ap[i]), 4),
                }
            )

        return {
            "map50": round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
            "mp": round(float(metrics.box.mp), 4),
            "mr": round(float(metrics.box.mr), 4),
            "class_metrics": class_metrics,
        }

    # ────────────────────────────── Per-Image Predict ────────────────────────────

    def run_predict(self, dataset_path: str, conf: float = 0.25) -> List[dict]:
        """Run model.predict() on every image and return per-image result list."""
        image_dir = self._find_image_dir(dataset_path)
        if image_dir is None:
            raise ValueError(f"No images found in: {dataset_path}")

        raw_results = self.model.predict(
            source=str(image_dir),
            conf=conf,
            verbose=False,
            stream=False,
        )

        results_list = []
        for result in raw_results:
            img_path = Path(result.path)
            detections: List[dict] = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for j in range(len(boxes)):
                    cls_id = int(boxes.cls[j])
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    conf_val = float(boxes.conf[j])
                    xyxy = boxes.xyxy[j].tolist()

                    detections.append(
                        {
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf_val, 4),
                            "x1": round(xyxy[0], 1),
                            "y1": round(xyxy[1], 1),
                            "x2": round(xyxy[2], 1),
                            "y2": round(xyxy[3], 1),
                        }
                    )

            # Aggregate per class
            class_counts: Dict[str, int] = {}
            class_confs: Dict[str, List[float]] = {}
            for det in detections:
                cn = det["class_name"]
                class_counts[cn] = class_counts.get(cn, 0) + 1
                class_confs.setdefault(cn, []).append(det["confidence"])

            results_list.append(
                {
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "total_detections": len(detections),
                    "class_counts": class_counts,
                    "class_avg_conf": {
                        k: round(sum(v) / len(v), 4) for k, v in class_confs.items()
                    },
                    "class_max_conf": {
                        k: round(max(v), 4) for k, v in class_confs.items()
                    },
                    "detections": detections,
                }
            )

        return results_list

    # ─────────────────────────────── Annotation ──────────────────────────────────

    # Distinct BGR-converted palette (stored as RGB tuples)
    _PALETTE = [
        (220, 50,  50),  # red
        (50,  205, 50),  # green
        (30,  120, 255), # blue
        (255, 200, 0),   # yellow
        (200, 0,  200),  # magenta
        (0,   210, 210), # cyan
        (255, 140, 0),   # orange
        (128, 0,  255),  # purple
        (0,   160, 80),  # dark green
        (255, 80,  150), # pink
        (80,  200, 255), # light blue
        (200, 255, 80),  # lime
        (160, 80,  0),   # brown
    ]

    def get_annotated_image(
        self, image_path: str, detections: List[dict]
    ) -> Optional[Image.Image]:
        """Draw bounding boxes + labels on image, return PIL Image (RGB)."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        font_scale = max(0.4, min(w, h) / 1000)
        thickness = max(1, int(min(w, h) / 500))

        for det in detections:
            cls_id = det["class_id"]
            color = self._PALETTE[cls_id % len(self._PALETTE)]
            x1 = max(0, int(det["x1"]))
            y1 = max(0, int(det["y1"]))
            x2 = min(w - 1, int(det["x2"]))
            y2 = min(h - 1, int(det["y2"]))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness + 1)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            label_y = max(y1 - 4, th + 8)
            cv2.rectangle(
                img,
                (x1, label_y - th - 6),
                (x1 + tw + 6, label_y + 2),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1 + 3, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return Image.fromarray(img)
