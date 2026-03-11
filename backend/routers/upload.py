import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/api/upload", tags=["upload"])

_UPLOAD_ROOT = Path(tempfile.gettempdir()) / "model_eval_uploads"
_UPLOAD_ROOT.mkdir(exist_ok=True)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _find_dataset_root(base: Path) -> Path:
    """
    Walk the extracted tree and return the dataset root — the directory whose
    structure matches a known YOLO layout (root/images/, root/val/images/, etc.).
    Falls back to the first directory that directly contains image files.
    """
    # Check standard YOLO subdirectory layouts (deepest-first so we catch nested zips)
    for layout in ("val/images", "test/images", "images"):
        parts = layout.split("/")
        for candidate in sorted(base.rglob(parts[-1]), key=lambda p: len(p.parts)):
            if not candidate.is_dir():
                continue
            # Verify this directory actually holds images
            if any(f.suffix.lower() in _IMAGE_EXTS for f in candidate.iterdir() if f.is_file()):
                # Walk UP by the number of parts to reach the dataset root
                root = candidate
                for _ in parts:
                    root = root.parent
                return root

    # Fallback: any directory that directly contains image files
    for dirpath in sorted(base.rglob("*"), key=lambda p: len(p.parts)):
        if dirpath.is_dir():
            if any(f.suffix.lower() in _IMAGE_EXTS for f in dirpath.iterdir() if f.is_file()):
                return dirpath

    return base


@router.post("/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Receive a .zip of a YOLO dataset, extract it, return the dataset root path."""
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Only .zip files accepted.")
    tmp = Path(tempfile.mkdtemp(dir=_UPLOAD_ROOT))
    zip_path = tmp / "dataset.zip"
    zip_path.write_bytes(await file.read())
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    zip_path.unlink()

    dataset_path = _find_dataset_root(tmp)
    return {"dataset_path": str(dataset_path)}
