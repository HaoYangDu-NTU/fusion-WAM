"""Record local artifacts without importing a policy or simulator."""
import argparse
import hashlib
import json
from pathlib import Path


def hash_file(path):
    with open(path, "rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def checkpoint_manifest(path):
    root = Path(path).resolve(strict=True)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    if not files or any(p.suffix in (".part", ".tmp") for p in files):
        raise ValueError("Checkpoint is empty or contains partial files")
    return {"checkpoint": str(root), "files": [
        {"path": str(p.relative_to(root)) if root.is_dir() else p.name,
         "bytes": p.stat().st_size, "sha256": hash_file(p)}
        for p in files
    ]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = checkpoint_manifest(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(manifest, f, indent=2)
