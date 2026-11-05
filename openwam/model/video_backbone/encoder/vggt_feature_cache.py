"""Content-addressed, process-safe frozen stereo features; no learned projections.

The exact preprocessed ordered RGB pair identifies a timestamp. This permits
reuse across overlapping windows without assuming frame-index decoding is
identical (random augmentations instead produce different keys).
"""
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class StereoFeatureCache:
    def __init__(self, directory, identity):
        self.identity = dict(identity)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.directory = Path(directory) / digest
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0
        # Namespace separation prevents reuse after changes to weights or processing.
        manifest = self.directory / "identity.json"
        with (self.directory / ".manifest.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if manifest.exists():
                if json.loads(manifest.read_text()) != identity:
                    raise ValueError("Cache identity mismatch")
            else:
                manifest.write_text(json.dumps(identity, indent=2) + "\n")

    def key(self, pair):
        pair = pair.detach().cpu().contiguous()
        descriptor = str((tuple(pair.shape), str(pair.dtype))).encode()
        return hashlib.sha256(descriptor + pair.view(torch.uint8).numpy().tobytes()).hexdigest()

    def get_or_encode(self, pair, encode):
        key = self.key(pair)
        directory = self.directory / key[:2]
        directory.mkdir(exist_ok=True)
        target = directory / (key + ".safetensors")
        with (directory / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                self.hits += 1
                result = load_file(target)["patches"]
            else:
                self.misses += 1
                result = encode(pair).detach().cpu().contiguous()
                if result.dtype not in (torch.float16, torch.bfloat16):
                    raise ValueError("Frozen features must be FP16/BF16; do not silently quantize")
                descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
                os.close(descriptor)
                try:
                    save_file({"patches": result}, temporary)
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        return result.to(device=pair.device, dtype=pair.dtype)


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
