"""Run a short, model-free installation check using upstream episodes."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from .evaluate import ARTIFACTS, ENVIRONMENTS, OFFICIAL, PROTOCOL, git_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if git_head(OFFICIAL) != PROTOCOL["molmospaces_commit"]:
        raise RuntimeError("MolmoSpaces commit differs from protocol.json")
    env = os.environ.copy()
    env.setdefault("MLSPACES_ASSETS_DIR", str(ARTIFACTS / "assets"))
    env.setdefault("MLSPACES_CACHE_DIR", str(ARTIFACTS / "resource-cache"))
    env.update(MLSPACES_FORCE_INSTALL="False", WANDB_MODE="disabled")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
    inventory = {}
    for name, (backend, directory, *_) in PROTOCOL["benchmarks"].items():
        path = Path(env["MLSPACES_ASSETS_DIR"]) / "benchmarks" / directory
        episodes = json.loads((path / "benchmark.json").read_text())
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"Empty or invalid episode list for {name}")
        inventory[name] = len(episodes)
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2))
    for name in ("MS-Pick", "MB-Pick-Filament"):
        backend, directory = PROTOCOL["benchmarks"][name]
        python = ENVIRONMENTS / (".venv-" + backend) / "bin/python"
        command = [str(python), "-m", "molmo_spaces.evaluation.eval_main",
                   "molmo_spaces.evaluation.configs.evaluation_configs:DummyBenchmarkEvalConfig",
                   "--benchmark_dir", str(Path(env["MLSPACES_ASSETS_DIR"]) / "benchmarks" / directory),
                   "--idx", "0", "--task_horizon_steps", "2", "--num_workers", "1",
                   "--no_wandb", "--output_dir", str(out / name)]
        if backend == "filament":
            command.append("--use-filament")
        log = out / (name + ".log")
        with log.open("w") as stream:
            subprocess.run(command, env=env, cwd=OFFICIAL, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        import re
        if not re.search(r"Evaluation complete: [01]/1 successful", log.read_text()):
            raise RuntimeError(f"Expected one completed episode in {log}")
        print(f"{name}: one two-step dummy-policy smoke completed", flush=True)


if __name__ == "__main__":
    main()
