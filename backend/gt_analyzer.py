"""
gt_analyzer.py  –  Ground Truth (GT) Loading & IoU Matching
Loads YOLO-format .txt label files and matches them against model predictions.
Enables per-image TP / FP / FN computation without running model.val().

YOLO label format (per line):
    class_id  cx  cy  width  height   (all values normalised 0-1)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_label_path(image_path: Path) -> Optional[Path]:
    """
    Search common YOLO dataset structures for the .txt label file
    that corresponds to a given image file.

    Supported layouts:
        root/images/img.jpg  →  root/labels/img.txt
        root/val/images/     →  root/val/labels/
        root/img.jpg         →  root/img.txt   (flat)
    """
    stem = image_path.stem
    candidates = [
        image_path.parent.parent / "labels"  / f"{stem}.txt",  # sibling labels/
        image_path.parent        / "labels"  / f"{stem}.txt",  # nested labels/
        image_path.parent                    / f"{stem}.txt",  # flat (same dir)
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_gt_boxes(
    image_path: Path,
    class_names: Dict[int, str],
    img_w: int,
    img_h: int,
) -> List[dict]:
    """
    Load YOLO label file and return GT boxes as absolute xyxy dicts.
    Returns [] when no label file is found (unlabelled image).

    Supports both numeric-id format  (0 cx cy w h)
    and named-class format           (person cx cy w h).
    """
    label_path = find_label_path(image_path)
    if label_path is None:
        return []

    # Reverse map for named labels: lowercase name → class_id
    name_to_id: Dict[str, int] = {v.lower(): k for k, v in class_names.items()}

    gt_boxes: List[dict] = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, nw, nh = (float(x) for x in parts[1:5])
            except ValueError:
                # First token is a class name (labels_with_name format)
                cls_id = name_to_id.get(parts[0].lower())
                if cls_id is None:
                    continue  # unknown class name — skip
                try:
                    cx, cy, nw, nh = (float(x) for x in parts[1:5])
                except ValueError:
                    continue

            x1 = (cx - nw / 2) * img_w
            y1 = (cy - nh / 2) * img_h
            x2 = (cx + nw / 2) * img_w
            y2 = (cy + nh / 2) * img_h
            bw, bh = x2 - x1, y2 - y1

            gt_boxes.append({
                "class_id":   cls_id,
                "class_name": class_names.get(cls_id, f"class_{cls_id}"),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width":   bw,
                "height":  bh,
                "area":    bw * bh,
                "cx_norm": cx,
                "cy_norm": cy,
            })
    return gt_boxes


def compute_iou(a: dict, b: dict) -> float:
    """IoU between two dicts containing x1, y1, x2, y2 keys."""
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    gt_boxes: List[dict],
    pred_boxes: List[dict],
    iou_thresh: float = 0.5,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Greedy IoU matching (highest-confidence predictions first).
    Only boxes of the SAME class are matched against each other.

    Returns:
        tp_pairs  – list of {'pred': ..., 'gt': ..., 'iou': float}
        fp_preds  – unmatched predictions  (False Positives)
        fn_gts    – unmatched GT boxes     (False Negatives / Misses)
    """
    sorted_preds = sorted(
        enumerate(pred_boxes),
        key=lambda x: x[1].get("confidence", 0),
        reverse=True,
    )
    matched_gt   = set()
    matched_pred = set()
    tp_pairs: List[dict] = []

    for pred_i, pred in sorted_preds:
        best_iou, best_gt_i = 0.0, -1
        for gt_i, gt in enumerate(gt_boxes):
            if gt_i in matched_gt:
                continue
            if gt["class_id"] != pred["class_id"]:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_gt_i = iou, gt_i

        if best_iou >= iou_thresh and best_gt_i >= 0:
            tp_pairs.append({"pred": pred, "gt": gt_boxes[best_gt_i], "iou": round(best_iou, 4)})
            matched_gt.add(best_gt_i)
            matched_pred.add(pred_i)

    fp_preds = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred]
    fn_gts   = [gt_boxes[i]  for i in range(len(gt_boxes))  if i not in matched_gt]
    return tp_pairs, fp_preds, fn_gts
