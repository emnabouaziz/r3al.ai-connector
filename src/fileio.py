"""Small, deterministic file-writing helpers."""

import os
from pathlib import Path


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + rename (no partial files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    """UTF-8, LF line endings on every platform (byte-identical reruns)."""
    write_bytes_atomic(path, text.encode("utf-8"))
