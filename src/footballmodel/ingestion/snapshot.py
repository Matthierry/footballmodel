from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class IngestionStatus:
    status: str
    changed: bool
    alert: bool
    detail: str


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"

    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_rerun(last_hash_path: str | Path, new_hash: str) -> bool:
    p = Path(last_hash_path)
    if not p.exists():
        return True
    old_hash = p.read_text(encoding="utf-8").strip()
    return old_hash != new_hash


def run_guarded_ingestion(
    source_file: str | Path,
    last_hash_path: str | Path,
    runner: Callable[[], None],
) -> IngestionStatus:
    new_hash = sha256_file(source_file)
    if not should_rerun(last_hash_path, new_hash):
        return IngestionStatus(status="skipped", changed=False, alert=False, detail="No source change")

    try:
        runner()
    except Exception as exc:  # intentionally broad: this is a run guard
        return IngestionStatus(status="failed", changed=True, alert=True, detail=str(exc))

    Path(last_hash_path).parent.mkdir(parents=True, exist_ok=True)
    Path(last_hash_path).write_text(new_hash, encoding="utf-8")
    return IngestionStatus(status="ran", changed=True, alert=False, detail="Pipeline executed")
