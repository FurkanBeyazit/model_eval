"""
evaluator.py  –  YOLO Model Evaluator
Handles model loading, validation (model.val), per-image prediction (model.predict),
threshold curve analysis, and image annotation.
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import tempfile
from PIL import Image
import gt_analyzer
import io

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Object size thresholds (COCO standard, in pixels²)
SIZE_SMALL  = 32 * 32    # < 1 024
SIZE_MEDIUM = 96 * 96    # < 9 216
# Large = >= 9 216


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
            self.class_names = self.model.names
            names_preview = ", ".join(
                f"{k}:{v}" for k, v in sorted(self.class_names.items())
            )
            return True, f"{len(self.class_names)} classes — {names_preview}"
        except Exception as exc:
            self.model = None
            return False, f"Error loading model: {exc}"

    # ────────────────────────────────── Helpers ──────────────────────────────────

    def _find_image_dir(self, dataset_path: str) -> Optional[Path]:
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
        return root

    def _create_yaml(self, dataset_path: str) -> str:
        root = Path(dataset_path.strip().strip('"').strip("'")).resolve()
        for rel in ("val/images", "test/images", "images", "val", "."):
            if (root / rel).exists():
                val_rel = rel
                break
        else:
            val_rel = "."
        # Forward slashes avoid PyYAML corrupting Windows backslash paths.
        # Write as plain text — no yaml.dump — so there is zero serialisation ambiguity.
        root_str = str(root).replace("\\", "/")
        names_lines = "\n".join(
            f"  {k}: {v}" for k, v in sorted(self.class_names.items(), key=lambda x: int(x[0]))
        )
        yaml_text = (
            f"path: {root_str}\n"
            f"train: {val_rel}\n"
            f"val: {val_rel}\n"
            f"nc: {len(self.class_names)}\n"
            f"names:\n{names_lines}\n"
        )
        yaml_path = Path(tempfile.gettempdir()) / "model_eval_temp.yaml"
        yaml_path.write_text(yaml_text, encoding="utf-8")
        print(f"[eval] yaml → {yaml_path}\n{yaml_text}")   # visible in backend console
        return str(yaml_path)

    # ──────────────────────────────── Validation ─────────────────────────────────

    def _scan_labels(self, dataset_path: str) -> tuple:
        """
        Scan YOLO label (.txt) files to count Images and Instances per class.
        Returns: (img_by_cls, inst_by_cls, total_images, total_instances)

        Label directory search order (most specific first):
          image_dir = root/val/images  →  tries root/val/labels, root/labels/val, root/labels
          image_dir = root/val         →  tries root/val/labels, root/labels/val, root/labels
          image_dir = root/images      →  tries root/labels, then root itself
          image_dir = root             →  tries root/labels, then root itself
        """
        root = Path(dataset_path.strip().strip('"').strip("'"))

        # Total image count from image directory
        image_dir = self._find_image_dir(dataset_path)
        total_images = (
            sum(1 for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if image_dir and image_dir.exists() else 0
        )

        # Build ordered list of candidate label directories, most specific first.
        # Critical: root/labels/ often contains ALL splits (train+val+test).
        # We must prefer the split-specific directory to avoid counting train labels.
        candidates: list[Path] = []
        if image_dir:
            if image_dir.name.lower() == "images":
                # root/val/images → root/val/labels (sibling of images inside val/)
                candidates.append(image_dir.parent / "labels")
                # root/val/images → root/labels/val (YOLO alternate layout)
                candidates.append(root / "labels" / image_dir.parent.name)
                # Flat: .txt files in same dir as images
                if any(image_dir.glob("*.txt")):
                    candidates.append(image_dir)
            else:
                # root/val → root/val/labels (sub-dir called labels)
                candidates.append(image_dir / "labels")
                # root/val → root/labels/val (parallel tree)
                candidates.append(root / "labels" / image_dir.name)
                # Flat: .txt alongside images
                if any(image_dir.glob("*.txt")):
                    candidates.append(image_dir)
            # Last resort: root/labels (all splits mixed — least preferred)
            candidates.append(root / "labels")

        labels_dir = None
        for c in candidates:
            if c.exists() and any(c.glob("*.txt")):
                labels_dir = c
                break

        if labels_dir is None:
            return {}, {}, total_images, 0, -1

        img_by_cls:  dict = {}
        inst_by_cls: dict = {}
        total_instances = 0
        max_cls_id = -1

        # Reverse map for named labels: lowercase name → class_id
        name_to_id = {v.lower(): k for k, v in self.class_names.items()}

        for label_file in sorted(labels_dir.glob("*.txt")):
            classes_seen: set = set()
            try:
                lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(parts[0])
                except (ValueError, IndexError):
                    # Named-class format (e.g. "person 0.5 0.5 0.3 0.4")
                    cls_id = name_to_id.get(parts[0].lower() if parts else "")
                    if cls_id is None:
                        continue
                try:
                    if cls_id > max_cls_id:
                        max_cls_id = cls_id
                    cls_name = self.class_names.get(cls_id, f"cls_{cls_id}")
                    inst_by_cls[cls_name] = inst_by_cls.get(cls_name, 0) + 1
                    classes_seen.add(cls_name)
                    total_instances += 1
                except (TypeError, IndexError):
                    continue
            for cls_name in classes_seen:
                img_by_cls[cls_name] = img_by_cls.get(cls_name, 0) + 1

        return img_by_cls, inst_by_cls, total_images, total_instances, max_cls_id

    def _print_val_table(
        self,
        total_imgs: int, total_inst: int,
        mp: float, mr: float, map50: float, map_val: float,
        class_metrics: list,
    ) -> None:
        """Print a YOLO-style validation results table to the terminal."""
        W = 88
        print(f"\n{'Class':>22}{'Images':>11}{'Instances':>11}"
              f"{'P':>11}{'R':>11}{'mAP50':>11}{'mAP50-95':>11}")
        print("─" * W)
        ti   = str(total_imgs) if total_imgs else "─"
        ti_i = str(total_inst) if total_inst else "─"
        print(f"{'all':>22}{ti:>11}{ti_i:>11}"
              f"{mp:>11.3g}{mr:>11.3g}{map50:>11.3g}{map_val:>11.3g}")
        for m in class_metrics:
            imgs  = str(m["Images"])    if isinstance(m["Images"],    int) else ""
            insts = str(m["Instances"]) if isinstance(m["Instances"], int) else ""
            print(f"{m['class']:>22}{imgs:>11}{insts:>11}"
                  f"{m['Precision']:>11.3g}{m['Recall']:>11.3g}"
                  f"{m['mAP50']:>11.3g}{m['mAP50-95']:>11.3g}")
        print("─" * W + "\n")

    def run_validation(self, dataset_path: str, conf: float = 0.25, iou: float = 0.45) -> dict:
        """Run model.val() — returns metrics + confusion matrix.

        Class-count mismatch handling:
          If the dataset contains label class IDs >= model.nc, we temporarily extend
          model.names with dummy entries so YOLO's ConfusionMatrix and print_results
          don't crash. Names are always restored via try/finally.
        """
        # 1. Scan labels — get Images/Instances counts AND max class id in dataset
        _img_by_cls, _inst_by_cls, _total_images, _total_instances, _max_cls_id = \
            self._scan_labels(dataset_path)

        # 2. Detect class-count mismatch and extend nc in yaml if needed.
        # We only update self.class_names (used by _create_yaml).
        # self.model.names is intentionally NOT touched — it is a read-only
        # property in some PyTorch versions and we don't need it:
        #   verbose=False → print_results (which reads model.names) is skipped
        #   plots=True    → plot_mc_curve only iterates ap_class_index (model's own
        #                    classes 0..nc-1), so extended class ids never appear there
        _model_nc       = len(self.class_names)
        _original_names = None
        if _max_cls_id >= _model_nc:
            _original_names = dict(self.class_names)
            extended = dict(self.class_names)
            for cid in range(_model_nc, _max_cls_id + 1):
                extended[cid] = f"cls_{cid}"
            self.class_names = extended   # _create_yaml uses this for nc
            print(f"[eval] class mismatch: model nc={_model_nc}, "
                  f"dataset max_cls_id={_max_cls_id} → yaml nc={len(extended)}")

        _has_mismatch = _max_cls_id >= _model_nc
        try:
            yaml_file = self._create_yaml(dataset_path)  # nc = len(self.class_names)
            metrics = self.model.val(
                data=yaml_file,
                conf=conf,
                iou=iou,
                split="val",
                verbose=False,       # we print our own table via _print_val_table
                plots=not _has_mismatch,  # mismatch: skip process_batch (IndexError)
                cache=False,         # prevent stale .cache files from causing wrong nc
                workers=0,           # disable multiprocessing (Windows pickle compat)
            )
        finally:
            # Always restore — even if val() raises
            if _original_names is not None:
                self.class_names = _original_names

        # Per-class metrics — skip extended (mismatch) classes so they don't
        # pollute the "all" averages. A 10-class model has no knowledge of
        # cls_10/11/12; including them as mAP=0 would unfairly drag down "all".
        ap_idx = (
            metrics.box.ap_class_index.tolist()
            if hasattr(metrics.box.ap_class_index, "tolist")
            else list(metrics.box.ap_class_index)
        )
        class_metrics = []
        for i, cls_id in enumerate(ap_idx):
            if int(cls_id) >= _model_nc:
                continue   # skip extended dummy classes
            cls_name = metrics.names.get(int(cls_id), f"class_{cls_id}")
            class_metrics.append({
                "class":      cls_name,
                "class_id":   int(cls_id),
                "Images":     _img_by_cls.get(cls_name, ""),
                "Instances":  _inst_by_cls.get(cls_name, ""),
                "Precision":  round(float(metrics.box.p[i]),    4),
                "Recall":     round(float(metrics.box.r[i]),    4),
                "F1":         round(float(metrics.box.f1[i]),   4),
                "mAP50":      round(float(metrics.box.ap50[i]), 4),
                "mAP50-95":   round(float(metrics.box.ap[i]),   4),
            })

        # Recompute "all" averages from model's own classes only (mismatch-safe)
        if class_metrics:
            _mp     = sum(m["Precision"] for m in class_metrics) / len(class_metrics)
            _mr     = sum(m["Recall"]    for m in class_metrics) / len(class_metrics)
            _map50  = sum(m["mAP50"]     for m in class_metrics) / len(class_metrics)
            _map    = sum(m["mAP50-95"]  for m in class_metrics) / len(class_metrics)
        else:
            _mp = float(metrics.box.mp)
            _mr = float(metrics.box.mr)
            _map50 = float(metrics.box.map50)
            _map   = float(metrics.box.map)

        # Print YOLO-style table to terminal
        self._print_val_table(
            _total_images, _total_instances,
            _mp, _mr, _map50, _map,
            class_metrics,
        )

        # Confusion Matrix
        cm_data = None
        try:
            cm_matrix = metrics.confusion_matrix.matrix   # shape (nc+1, nc+1)
            nc = len(self.class_names)
            labels = [self.class_names.get(i, f"class_{i}") for i in range(nc)] + ["background"]
            cm_data = {
                "matrix": cm_matrix.tolist(),
                "labels": labels,
            }
        except Exception:
            pass

        return {
            "map50":            round(_map50, 4),
            "map50_95":         round(_map,   4),
            "mp":               round(_mp,    4),
            "mr":               round(_mr,    4),
            "total_images":     _total_images,
            "total_instances":  _total_instances,
            "class_metrics":    class_metrics,
            "confusion_matrix": cm_data,
        }

    # ──────────────────────────── GT-Aware Predict ───────────────────────────────

    def run_predict_with_gt(
        self,
        dataset_path: str,
        conf: float = 0.25,
        iou_thresh: float = 0.5,
    ) -> Tuple[List[dict], List[dict], bool]:
        """
        Run predict at conf=0.01, then:
          1. Load GT labels for each image (if available).
          2. Match predictions to GT via IoU (greedy, same-class).
          3. Return:
             - predict_results  (same structure as run_predict, enriched with GT counts)
             - raw_data         (per-image dicts with all preds + GT, for threshold curve)
             - has_gt           (True if at least one image has a label file)

        Each item in predict_results gains extra keys:
            gt_counts, tp, fp, fn, match_rate
        Each item in raw_data:
            image_path, image_name, img_width, img_height,
            preds (all predictions at conf=0.01),
            gt_boxes,
            tp_pairs, fp_preds, fn_gts  (at the requested iou_thresh)
        """
        image_dir = self._find_image_dir(dataset_path)
        if image_dir is None:
            raise ValueError(f"No images found in: {dataset_path}")

        # Run at very low conf so we keep everything for threshold sweep
        low_conf_results = self.model.predict(
            source=str(image_dir),
            conf=0.01,
            verbose=False,
            stream=False,
        )

        predict_results: List[dict] = []
        raw_data: List[dict] = []
        has_gt = False

        for result in low_conf_results:
            img_path = Path(result.path)
            img_h, img_w = result.orig_shape[:2]

            # ── Build prediction list ──────────────────────────────────────────
            all_preds: List[dict] = []
            if result.boxes is not None and len(result.boxes) > 0:
                for j in range(len(result.boxes)):
                    cls_id   = int(result.boxes.cls[j])
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    conf_val = float(result.boxes.conf[j])
                    xyxy     = result.boxes.xyxy[j].tolist()
                    bw = xyxy[2] - xyxy[0]
                    bh = xyxy[3] - xyxy[1]
                    cx_norm = round((xyxy[0] + xyxy[2]) / (2 * img_w), 4)
                    cy_norm = round((xyxy[1] + xyxy[3]) / (2 * img_h), 4)
                    all_preds.append({
                        "class_id":   cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf_val, 4),
                        "x1": round(xyxy[0], 1), "y1": round(xyxy[1], 1),
                        "x2": round(xyxy[2], 1), "y2": round(xyxy[3], 1),
                        "width":   round(bw, 1),
                        "height":  round(bh, 1),
                        "area":    round(bw * bh, 1),
                        "cx_norm": cx_norm,
                        "cy_norm": cy_norm,
                    })

            # ── Load GT labels ─────────────────────────────────────────────────
            gt_boxes = gt_analyzer.load_gt_boxes(img_path, self.class_names, img_w, img_h)
            if gt_boxes:
                has_gt = True

            # ── Filter preds at requested conf for predict_results ─────────────
            filtered_preds = [p for p in all_preds if p["confidence"] >= conf]

            # ── Match at requested conf threshold ──────────────────────────────
            tp_pairs, fp_preds, fn_gts = gt_analyzer.match_detections(
                gt_boxes, filtered_preds, iou_thresh
            )

            # ── Build GT count per class ───────────────────────────────────────
            gt_counts: Dict[str, int] = {}
            for g in gt_boxes:
                gt_counts[g["class_name"]] = gt_counts.get(g["class_name"], 0) + 1

            # ── class_counts / confs for filtered preds ────────────────────────
            class_counts: Dict[str, int] = {}
            class_confs:  Dict[str, List[float]] = {}
            for det in filtered_preds:
                cn = det["class_name"]
                class_counts[cn] = class_counts.get(cn, 0) + 1
                class_confs.setdefault(cn, []).append(det["confidence"])

            match_rate = (
                round(len(tp_pairs) / len(gt_boxes), 4) if gt_boxes else None
            )

            predict_results.append({
                "image_path":        str(img_path),
                "image_name":        img_path.name,
                "img_width":         img_w,
                "img_height":        img_h,
                "total_detections":  len(filtered_preds),
                "class_counts":      class_counts,
                "class_avg_conf":    {k: round(sum(v) / len(v), 4) for k, v in class_confs.items()},
                "class_max_conf":    {k: round(max(v), 4)          for k, v in class_confs.items()},
                "detections":        filtered_preds,
                # GT-aware extras
                "gt_counts":         gt_counts,
                "tp":                len(tp_pairs),
                "fp":                len(fp_preds),
                "fn":                len(fn_gts),
                "match_rate":        match_rate,
                # Per-image match details (for comparison image)
                "tp_pairs":          tp_pairs,
                "fp_preds":          fp_preds,
                "fn_gts":            fn_gts,
            })

            raw_data.append({
                "image_path":  str(img_path),
                "image_name":  img_path.name,
                "img_width":   img_w,
                "img_height":  img_h,
                "preds":       all_preds,   # all at conf=0.01
                "gt_boxes":    gt_boxes,
            })

        return predict_results, raw_data, has_gt

    # ──────────────────────────── Threshold Curve Analysis ───────────────────────

    def run_threshold_analysis(
        self,
        raw_data: List[dict],
        iou_thresh: float = 0.5,
    ) -> List[dict]:
        """
        Compute P / R / F1 at each confidence threshold using raw_data
        (output of run_predict_with_gt).

        If no GT labels are present (has_gt=False), falls back to counting
        detections only (no P/R/F1).
        """
        thresholds = [round(t / 100, 2) for t in range(10, 96, 5)]  # 0.10 … 0.95
        all_cls    = sorted(self.class_names.values())
        has_gt     = any(item["gt_boxes"] for item in raw_data)

        curve: List[dict] = []
        for thresh in thresholds:
            total_tp = total_fp = total_fn = 0
            cls_tp: Dict[str, int] = {c: 0 for c in all_cls}
            cls_fp: Dict[str, int] = {c: 0 for c in all_cls}
            cls_fn: Dict[str, int] = {c: 0 for c in all_cls}
            cls_det_counts: Dict[str, int] = {c: 0 for c in all_cls}
            total_dets = 0

            for item in raw_data:
                filtered = [p for p in item["preds"] if p["confidence"] >= thresh]
                total_dets += len(filtered)

                if has_gt:
                    tp_pairs, fp_preds, fn_gts = gt_analyzer.match_detections(
                        item["gt_boxes"], filtered, iou_thresh
                    )
                    total_tp += len(tp_pairs)
                    total_fp += len(fp_preds)
                    total_fn += len(fn_gts)
                    for pair in tp_pairs:
                        cn = pair["pred"]["class_name"]
                        if cn in cls_tp:
                            cls_tp[cn] += 1
                    for p in fp_preds:
                        cn = p["class_name"]
                        if cn in cls_fp:
                            cls_fp[cn] += 1
                    for g in fn_gts:
                        cn = g["class_name"]
                        if cn in cls_fn:
                            cls_fn[cn] += 1
                else:
                    for p in filtered:
                        cn = p["class_name"]
                        if cn in cls_det_counts:
                            cls_det_counts[cn] += 1

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            row: dict = {
                "Threshold": thresh,
                "Total":     total_dets if not has_gt else total_tp + total_fp,
            }
            if has_gt:
                row["TP"]        = total_tp
                row["FP"]        = total_fp
                row["FN"]        = total_fn
                row["Precision"] = round(precision, 4)
                row["Recall"]    = round(recall, 4)
                row["F1"]        = round(f1, 4)

            for cls in all_cls:
                if has_gt:
                    row[f"{cls}_TP"] = cls_tp[cls]
                    row[f"{cls}_FP"] = cls_fp[cls]
                    row[f"{cls}_FN"] = cls_fn[cls]
                else:
                    row[cls] = cls_det_counts[cls]

            curve.append(row)

        return curve

    def run_threshold_analysis_from_dir(self, dataset_path: str) -> List[dict]:
        """Legacy: runs predict at conf=0.01 without GT. Used as fallback."""
        image_dir = self._find_image_dir(dataset_path)
        if image_dir is None:
            return []

        raw_results = self.model.predict(
            source=str(image_dir), conf=0.01, verbose=False, stream=False,
        )
        raw_data = []
        for result in raw_results:
            img_h, img_w = result.orig_shape[:2]
            preds = []
            if result.boxes is not None and len(result.boxes) > 0:
                for j in range(len(result.boxes)):
                    cls_id = int(result.boxes.cls[j])
                    xyxy   = result.boxes.xyxy[j].tolist()
                    bw = xyxy[2] - xyxy[0]; bh = xyxy[3] - xyxy[1]
                    preds.append({
                        "class_id":   cls_id,
                        "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                        "confidence": float(result.boxes.conf[j]),
                        "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
                        "width": bw, "height": bh, "area": bw * bh,
                        "cx_norm": (xyxy[0] + xyxy[2]) / (2 * img_w),
                        "cy_norm": (xyxy[1] + xyxy[3]) / (2 * img_h),
                    })
            raw_data.append({
                "image_path": result.path, "image_name": Path(result.path).name,
                "img_width": img_w, "img_height": img_h,
                "preds": preds, "gt_boxes": [],
            })
        return self.run_threshold_analysis(raw_data)

    # ─────────────────────────────── Annotation ──────────────────────────────────

    # GT comparison colours (BGR for OpenCV)
    _GT_COLOUR = (0,   210,   0)   # green  – TP matched GT box
    _FP_COLOUR = (50,   50, 220)   # red    – False Positive prediction
    _FN_COLOUR = (0,  165, 255)    # orange – False Negative (missed GT)

    def get_comparison_image(
        self,
        image_path: str,
        tp_pairs: List[dict],
        fp_preds: List[dict],
        fn_gts:   List[dict],
    ) -> Optional[Image.Image]:
        """
        Draw a GT-vs-prediction comparison overlay.
          Green  (solid)  – TP: GT box that was matched
          Lime   (dashed) – TP: matching prediction box
          Red             – FP: prediction with no GT match
          Orange          – FN: GT box that was missed
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        font_scale = max(0.4, min(w, h) / 1500)
        thick = max(1, int(min(w, h) / 1000))

        def _draw_rect(det, color, dash=False):
            x1 = max(0, int(det["x1"])); y1 = max(0, int(det["y1"]))
            x2 = min(w - 1, int(det["x2"])); y2 = min(h - 1, int(det["y2"]))
            if dash:
                seg = 12
                for sx in range(x1, x2, seg * 2):
                    cv2.line(img, (sx, y1), (min(sx + seg, x2), y1), color, thick)
                    cv2.line(img, (sx, y2), (min(sx + seg, x2), y2), color, thick)
                for sy in range(y1, y2, seg * 2):
                    cv2.line(img, (x1, sy), (x1, min(sy + seg, y2)), color, thick)
                    cv2.line(img, (x2, sy), (x2, min(sy + seg, y2)), color, thick)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thick + 1)
            return x1, y1

        def _draw_label(x1, y1, color, label_text, offset_y=0):
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            ly = max(y1 - 4 - offset_y, th + 8)
            cv2.rectangle(img, (x1, ly - th - 6), (x1 + tw + 6, ly + 2), color, -1)
            cv2.putText(img, label_text, (x1 + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # TP: GT box (solid) with label, pred box (dashed) without label — same location,
        # two labels would overlap. IoU is included in the GT label.
        for pair in tp_pairs:
            x1, y1 = _draw_rect(pair["gt"], self._GT_COLOUR)
            _draw_label(x1, y1, self._GT_COLOUR, pair["gt"]["class_name"])

        for p in fp_preds:
            x1, y1 = _draw_rect(p, self._FP_COLOUR)
            _draw_label(x1, y1, self._FP_COLOUR,
                        f"FP {p['class_name']} {p['confidence']:.2f}")

        for g in fn_gts:
            x1, y1 = _draw_rect(g, self._FN_COLOUR)
            _draw_label(x1, y1, self._FN_COLOUR, f"FN {g['class_name']}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    _PALETTE = [
        (220, 50,  50), (50,  205, 50), (30,  120, 255),
        (255, 200,  0), (200,  0, 200), ( 0,  210, 210),
        (255, 140,  0), (128,  0, 255), ( 0,  160,  80),
        (255,  80, 150), (80, 200, 255), (200, 255,  80),
        (160,  80,   0),
    ]

    # ──────────────────────────────── Benchmark ──────────────────────────────────

    def run_benchmark(
        self,
        dataset_path: str,
        n_warmup: int = 20,
        n_measure: int = 50,
    ) -> dict:
        """Measure inference speed: batch=1 latency and batch=8 throughput.

        Uses torch.cuda.synchronize() so timings reflect actual GPU completion,
        not just CPU dispatch. Warm-up prevents cold-start CUDA kernel bias.
        """
        import time
        import torch

        image_dir = self._find_image_dir(dataset_path)
        if image_dir is None:
            raise ValueError(f"No images found in: {dataset_path}")

        images = sorted(
            str(f) for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            raise ValueError("No images found")

        use_cuda = torch.cuda.is_available()
        device   = "cuda" if use_cuda else "cpu"

        # Model metadata
        model_size_mb: Optional[float] = None
        params_m: Optional[float]      = None
        if self.model_path:
            try:
                model_size_mb = round(Path(self.model_path).stat().st_size / 1024 ** 2, 1)
            except Exception:
                pass
        try:
            params_m = round(
                sum(p.numel() for p in self.model.model.parameters()) / 1e6, 1
            )
        except Exception:
            pass

        # Repeat image list if dataset is smaller than n_measure
        factor     = (max(n_warmup, n_measure) // len(images)) + 2
        pool       = images * factor
        warmup_lst = pool[:n_warmup]
        meas_lst   = pool[:n_measure]

        # ── Warm-up ───────────────────────────────────────────────────────────
        self.model.predict(source=warmup_lst, conf=0.25, verbose=False, stream=False)
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # ── Batch = 1  (single-frame latency — CCTV real-time) ────────────────
        t0 = time.perf_counter()
        for img in meas_lst:
            self.model.predict(source=img, conf=0.25, verbose=False, stream=False)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_b1 = time.perf_counter() - t0

        inf_ms = round(elapsed_b1 * 1000 / len(meas_lst), 2)
        fps_b1 = round(len(meas_lst) / elapsed_b1, 1)
        vram_b1: Optional[float] = None
        if use_cuda:
            vram_b1 = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            torch.cuda.reset_peak_memory_stats()

        # ── Batch = 8  (multi-stream throughput) ─────────────────────────────
        batch_sz = 8
        batches  = [meas_lst[i: i + batch_sz] for i in range(0, len(meas_lst), batch_sz)]
        n_b8     = sum(len(b) for b in batches)

        t0 = time.perf_counter()
        for batch in batches:
            self.model.predict(source=batch, conf=0.25, verbose=False, stream=False)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_b8 = time.perf_counter() - t0

        fps_b8   = round(n_b8 / elapsed_b8, 1)
        vram_b8: Optional[float] = None
        if use_cuda:
            vram_b8 = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)

        return {
            "inf_ms":        inf_ms,
            "fps_b1":        fps_b1,
            "fps_b8":        fps_b8,
            "vram_b1_gb":    vram_b1,
            "vram_b8_gb":    vram_b8,
            "model_size_mb": model_size_mb,
            "params_m":      params_m,
            "n_images":      len(meas_lst),
            "n_warmup":      n_warmup,
            "device":        device,
        }

    def get_annotated_image(
        self, image_path: str, detections: List[dict], highlight_idx: int = -1
    ) -> Optional[Image.Image]:
        """
        Draw detections on image.
        highlight_idx: if >= 0, that detection gets a bright yellow halo border.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        font_scale = max(0.4, min(w, h) / 1500)
        thickness  = max(1, int(min(w, h) / 1000))

        for i, det in enumerate(detections):
            cls_id = det["class_id"]
            color  = self._PALETTE[cls_id % len(self._PALETTE)]
            x1 = max(0, int(det["x1"])); y1 = max(0, int(det["y1"]))
            x2 = min(w - 1, int(det["x2"])); y2 = min(h - 1, int(det["y2"]))

            # Highlighted detection: yellow halo drawn first (behind the box)
            if i == highlight_idx:
                halo = thickness + 5
                cv2.rectangle(img,
                              (max(0, x1 - halo), max(0, y1 - halo)),
                              (min(w - 1, x2 + halo), min(h - 1, y2 + halo)),
                              (255, 230, 0), halo)

            box_thick = (thickness + 3) if i == highlight_idx else (thickness + 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_y = max(y1 - 4, th + 8)
            cv2.rectangle(img, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 2), color, -1)
            cv2.putText(img, label, (x1 + 3, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        return Image.fromarray(img)
