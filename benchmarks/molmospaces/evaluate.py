"""Drive the pinned upstream evaluator and CSV exporter in their own environments."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from .provenance import hash_file

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / ".benchmark_runtime" / "molmospaces"
OFFICIAL = CODE / "molmospaces-official"
ENVIRONMENTS = CODE / "molmospaces"
OPENPI = CODE / "openpi-molmo-jointpos"
ARTIFACTS = CODE / "molmospaces-artifacts"
PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())


def git_head(repo):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def run_checked(command, log, env, cwd=OFFICIAL):
    with log.open("w") as stream:
        subprocess.run([str(x) for x in command], check=True, cwd=cwd, env=env,
                       stdout=stream, stderr=subprocess.STDOUT)


def evaluate(args):
    config = "benchmarks.molmospaces.sim_config:RemoteDroidEvalConfig"
    if git_head(OFFICIAL) != PROTOCOL["molmospaces_commit"]:
        raise RuntimeError("Expected the official MolmoSpaces v0.2.9 source checkout")
    if subprocess.check_output(["git", "-C", str(OFFICIAL), "diff", "HEAD", "--name-only"]):
        raise RuntimeError("Official MolmoSpaces source has tracked modifications")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_manifest = args.checkpoint_manifest.resolve()
    if not checkpoint_manifest.is_file():
        raise FileNotFoundError(checkpoint_manifest)
    provenance = {"mode": "smoke" if args.smoke else "full", "config": config,
                  "project_commit": git_head(PROJECT), "molmospaces_commit": git_head(OFFICIAL),
                  "checkpoint_manifest": str(checkpoint_manifest),
                  "checkpoint_manifest_sha256": hash_file(checkpoint_manifest),
                  "policy": args.policy_name, "workers": args.workers}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    env = os.environ.copy()
    env.setdefault("MLSPACES_ASSETS_DIR", str(ARTIFACTS / "assets"))
    env.setdefault("MLSPACES_CACHE_DIR", str(ARTIFACTS / "resource-cache"))
    env.setdefault("MLSPACES_FORCE_INSTALL", "False")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MOLMO_POLICY_PORT"] = str(args.port)
    selections = args.suite or list(PROTOCOL["benchmarks"])
    if len(set(selections)) != len(selections):
        raise ValueError("Duplicate suites")
    for name in selections:
        renderer, relative, *camera = PROTOCOL["benchmarks"][name]
        python = ENVIRONMENTS / (".venv-" + renderer) / "bin/python"
        suite_dir = out / name
        suite_dir.mkdir()
        benchmark = Path(env["MLSPACES_ASSETS_DIR"]) / "benchmarks" / relative
        if not benchmark.is_dir():
            raise FileNotFoundError(f"Official benchmark missing: {benchmark}")
        count = subprocess.check_output(
            [str(python), "-c",
             "import sys; from pathlib import Path; "
             "from molmo_spaces.evaluation.eval_main import load_all_episodes; "
             "print(len(load_all_episodes(Path(sys.argv[1]))))", str(benchmark)],
            env=env, cwd=OFFICIAL, text=True)
        expected = 1 if args.smoke else int(count.strip().splitlines()[-1])
        freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
        (suite_dir / "packages.txt").write_text(freeze)
        command = [python, "molmo_spaces/evaluation/eval_main.py", config, "--benchmark_dir",
                   benchmark, "--output_dir", suite_dir / "rollouts", "--num_workers",
                   args.workers, "--no_wandb"]
        if renderer == "filament":
            command += ["--use-filament"]
        if camera:
            command += ["--camera_names", *camera[0]]
        if args.smoke:
            command += ["--idx", 0]
        (suite_dir / "command.json").write_text(json.dumps([str(x) for x in command], indent=2))
        run_checked(command, suite_dir / "rollout.log", env)
        runs = list((suite_dir / "rollouts" / "RemoteDroidEvalConfig").iterdir())
        if len(runs) != 1:
            raise RuntimeError(f"Expected exactly one evaluator output for {name}")
        report = suite_dir / "official.csv"
        run_checked([python, "scripts/benchmarks/eval_to_csv.py", runs[0], args.policy_name,
                     "--success-condition", "both", "--output-csv", report],
                    suite_dir / "csv.log", env)
        with report.open() as f:
            rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
        total = next(row for row in rows if row["category"] == "OVERALL")
        if int(total["total"]) != expected:
            raise RuntimeError(f"{name} incomplete: {total['total']}/{expected}")
        metrics = {"suite": name, "episodes": expected, "oracle_sr_pct": float(total["oracle_rate_pct"]),
                   "end_sr_pct": float(total["success_rate_pct"]), "renderer": renderer}
        with (out / "metrics.jsonl").open("a") as f:
            print(json.dumps(metrics), file=f)
        print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", choices=PROTOCOL["benchmarks"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
