from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_gdrive_file_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    # Accept raw file id
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", s) and "http" not in s:
        return s
    # Common share URL formats:
    # - https://drive.google.com/file/d/<id>/view?...
    m = re.search(r"/file/d/([^/]+)/", s)
    if m:
        return m.group(1)
    # - https://drive.google.com/uc?id=<id>
    m = re.search(r"[?&]id=([^&]+)", s)
    if m:
        return m.group(1)
    raise ValueError(f"Could not parse Google Drive file id from: {url_or_id}")


def download_gdrive(url_or_id: str, out_path: str | Path, *, quiet: bool = False) -> str:
    """
    Download a Google Drive file (by share URL or file id) using gdown.

    Returns the output path as a string.
    """
    out_path = str(out_path)
    ensure_dir(Path(out_path).parent)
    file_id = _parse_gdrive_file_id(url_or_id)

    try:
        import gdown  # type: ignore
    except Exception as e:
        raise RuntimeError("gdown is required. Install with: pip install gdown") from e

    # Use id= to avoid URL parsing issues.
    url = f"https://drive.google.com/uc?id={file_id}"
    if not quiet:
        print(f"Downloading Google Drive file id={file_id} to {out_path} ...")
    gdown.download(url, out_path, quiet=quiet, fuzzy=True)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Download failed or produced empty file: {out_path}")
    return out_path


def safe_extract_archive(archive_path: str | Path, dest_dir: str | Path) -> str:
    """
    Extract .zip or .tar(.gz/.bz2/.xz) archives to dest_dir with basic path traversal checks.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    ensure_dir(dest_dir)

    def _is_within_directory(base: Path, target: Path) -> bool:
        base = base.resolve()
        target = target.resolve()
        return str(target).startswith(str(base))

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.infolist():
                target = dest_dir / member.filename
                if not _is_within_directory(dest_dir, target):
                    raise RuntimeError(f"Blocked zip path traversal attempt: {member.filename}")
            zf.extractall(dest_dir)
        return str(dest_dir)

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            for member in tf.getmembers():
                target = dest_dir / member.name
                if not _is_within_directory(dest_dir, target):
                    raise RuntimeError(f"Blocked tar path traversal attempt: {member.name}")
            tf.extractall(dest_dir)
        return str(dest_dir)

    raise ValueError(f"Unsupported archive format: {archive_path}")


@dataclass(frozen=True)
class DatasetLayout:
    dataset_root: str
    meta_path: str
    train_matrix_path: str
    val_matrix_path: str | None
    test_matrix_path: str | None
    review_path: str


def validate_dataset_layout(dataset_root: str) -> DatasetLayout:
    """
    Validates the expected dataset layout, returning resolved paths.
    """
    root = Path(dataset_root)
    meta_path = root / "meta.pkl"
    train_path = root / "train_matrix.npz"
    val_path = root / "val_matrix.npz"
    test_path = root / "test_matrix.npz"
    review_path = root / "user_item_texts" / "review.pkl"

    missing = []
    for p in [meta_path, train_path, review_path]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Dataset is missing required files:\n" + "\n".join(missing))

    return DatasetLayout(
        dataset_root=str(root),
        meta_path=str(meta_path),
        train_matrix_path=str(train_path),
        val_matrix_path=str(val_path) if val_path.exists() else None,
        test_matrix_path=str(test_path) if test_path.exists() else None,
        review_path=str(review_path),
    )


def build_dataset_manifest(dataset_root: str, *, include_optional: bool = True) -> dict[str, Any]:
    """
    Build a reproducibility manifest (metadata-only) for W&B logging.
    """
    layout = validate_dataset_layout(dataset_root)
    required = [
        layout.meta_path,
        layout.train_matrix_path,
        layout.review_path,
    ]
    if include_optional:
        if layout.val_matrix_path:
            required.append(layout.val_matrix_path)
        if layout.test_matrix_path:
            required.append(layout.test_matrix_path)

    files = []
    for p in required:
        files.append(
            {
                "path": os.path.relpath(p, dataset_root),
                "bytes": os.path.getsize(p),
                "sha256": sha256_file(p),
            }
        )

    manifest = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "files": files,
    }
    # Stable overall hash for quick identity checks
    manifest_json = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(manifest_json).hexdigest()
    return manifest


def save_json(obj: Any, path: str | Path) -> str:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return str(path)


