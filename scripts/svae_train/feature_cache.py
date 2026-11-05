"""Release only S-VAE feature file pages; never delete files or drop host caches."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def release_file_cache(path: Path, *, sync: bool = False) -> None:
    """Flush a completed write if requested, then advise eviction of clean pages."""
    if not hasattr(os, "posix_fadvise"):
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        if sync:
            os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _in_use(paths: set[str]) -> set[str]:
    """Keep features still mapped or open by another visible process in cache."""
    busy: set[str] = set()
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return busy
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for line in (proc / "maps").read_text(errors="replace").splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5] in paths:
                    busy.add(fields[5])
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if target in paths:
                        busy.add(target)
                except OSError:
                    pass  # Process/fd may exit between enumeration and readlink.
        except OSError:
            continue
    return busy


def release_feature_cache(features_dir: Path) -> dict:
    """Call after readers/workers close their mmaps, once per node."""
    result = {"released": 0, "skipped_in_use": 0, "errors": []}
    if not hasattr(os, "posix_fadvise"):
        logger.warning("[svae-cache] posix_fadvise unavailable; cache release skipped")
        return result
    paths = {str(p.resolve()) for p in features_dir.glob("features_rank*_part*.pt")}
    busy = _in_use(paths)
    result["skipped_in_use"] = len(busy)
    started = time.monotonic()
    logger.info("[svae-cache] releasing pages for %d inactive feature shards", len(paths - busy))
    for name in sorted(paths - busy):
        try:
            release_file_cache(Path(name), sync=True)
            result["released"] += 1
        except OSError as exc:
            result["errors"].append(f"{name}: {exc}")
    logger.info("[svae-cache] feature cache release: %s (%.2fs)", result, time.monotonic() - started)
    if busy or result["errors"]:
        logger.warning("[svae-cache] some feature pages could not be released: %s", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_dir", type=Path)
    args = parser.parse_args()
    result = release_feature_cache(args.features_dir)
    print(json.dumps(result))
    raise SystemExit(1 if result["errors"] else 0)
