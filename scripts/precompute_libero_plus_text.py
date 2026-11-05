"""Precompute exact Plus instruction modes with UMT5 or frozen HF VLMs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.libero_plus_instructions import task_instruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-path", type=Path, default=Path("/workspace/code/benchmarks/LIBERO-plus"))
    parser.add_argument("--libero-plus-python", default="/workspace/basics/miniconda3/envs/libero-plus/bin/python")
    from openwam.text_condition.precompute import add_encoder_arguments, write_caches

    add_encoder_arguments(parser)
    parser.add_argument("--catalog-dir", type=Path, help="Existing Plus cache root containing original_text/corrected_text manifests")
    parser.add_argument("--text-mode", choices=("both", "original_text", "corrected_text"), default="corrected_text")
    args = parser.parse_args()
    if args.catalog_dir:
        modes = ("original_text", "corrected_text") if args.text_mode == "both" else (args.text_mode,)
        catalogs = {mode: list(json.loads((args.catalog_dir / mode / "manifest.json").read_text())["prompts"])
                    for mode in modes}
        write_caches(catalogs, args, "libero-plus")
        return
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        raise ValueError("Use --catalog-dir for distributed precomputation")

    osmesa = ROOT / "third_party/osmesa/usr/lib/x86_64-linux-gnu"
    original = None
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix="openwam-plus-prompts-") as temporary:
            output = Path(temporary) / "tasks.json"
            env = {**os.environ, "PYTHONPATH": f"{ROOT}:{args.libero_path}",
                   "LIBERO_PLUS_PATH": str(args.libero_path),
                   "LIBERO_PLUS_CONFIG_ROOT": str(Path(temporary) / "config"),
                   "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
                   "LD_LIBRARY_PATH": str(osmesa) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            subprocess.run([args.libero_plus_python, "-m", "benchmarks.vla_harness.libero_plus_env",
                            "--describe", str(output), "--correct-instruction-suffix", "False"],
                           cwd=ROOT, env=env, check=True, timeout=180)
            original = json.loads(output.read_text())
    catalogs = {}
    for mode, correct in (("original_text", False), ("corrected_text", True)):
        if args.text_mode not in ("both", mode):
            continue
        catalogs[mode] = [task_instruction(SimpleNamespace(name=row["task_name"], language=row["language"]), correct)
                          for row in original]
    write_caches(catalogs, args, "libero-plus")


if __name__ == "__main__":
    main()
