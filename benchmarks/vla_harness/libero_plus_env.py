"""One fresh official LIBERO-plus simulator process per task; no harness dependency."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from multiprocessing.connection import Connection
from pathlib import Path

from benchmarks.libero_plus_instructions import parse_bool, task_instruction

OFFICIAL = Path(__file__).resolve().parents[1] / "libero-plus"


def official_modules():
    # This process never imports standard LIBERO. Preserve single_eval's own
    # script-local interface import and its exact seed-before-benchmark order.
    sys.path.insert(0, str(OFFICIAL))
    import single_eval

    cfg = single_eval._load_config(OFFICIAL / "policy_config.yml")
    single_eval._write_libero_config()
    single_eval._control_seed(int(cfg["seed"]))
    from libero.libero import benchmark

    expected = (single_eval._repo_root() / "libero/libero/benchmark/__init__.py").resolve()
    if Path(benchmark.__file__).resolve() != expected:
        raise RuntimeError(f"Wrong LIBERO import: {benchmark.__file__}; expected {expected}")
    return single_eval, cfg, benchmark


def serve(connection, suite, task_id, correct_instruction_suffix=True):
    env = None
    try:
        official, cfg, benchmark = official_modules()
        source = benchmark.get_benchmark_dict()[suite]()
        task = source.get_task(task_id)
        settle = official._parse_settle_action(cfg["settle_action"], 7)
        states = source.get_task_init_states(task_id)
        env = official._make_env_with_randomization_retries(task, cfg)
        if 21 <= getattr(env, "noise", 0) <= 30 or 41 <= getattr(env, "noise", 0) <= 50:
            # Import/JIT only for zoom/glass-noise tasks. Keep setup from consuming
            # the official episode RNG; the kernel draws from NumPy unchanged.
            import random
            import numpy as np
            rng, py_rng = np.random.get_state(), random.getstate()
            try:
                from benchmarks.vla_harness.libero_plus_noise import glass_blur, clipped_zoom, _shuffle_pixels
                from libero.libero.envs import env_wrapper

                if env.noise >= 41:
                    _shuffle_pixels(np.zeros((4, 4, 3), dtype=np.uint8),
                                    np.zeros((4, 2), dtype=np.int64), 1, 1)
                    env_wrapper.glass_blur = glass_blur
                else:
                    env_wrapper.clipped_zoom = clipped_zoom
            finally:
                np.random.set_state(rng)
                random.setstate(py_rng)
        # No env.seed(): Plus intentionally uses official_global RNG mode.
        obs = env.reset()
        if len(states):
            obs = env.set_init_state(states[0])
        for _ in range(int(cfg["settle_steps"])):
            obs, _, _, _ = env.step(settle)
        connection.send({"obs": obs, "language": task_instruction(task, correct_instruction_suffix), "task_name": task.name})
        import numpy as np

        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command != "step":
                raise ValueError(f"Unknown simulator command: {command}")
            # Python lists cross the NumPy 2.x parent / NumPy 1.x child boundary.
            action = np.asarray(payload, dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Expected finite 7D LIBERO action")
            connection.send({"result": env.step(action)})
    except (EOFError, BrokenPipeError):
        pass  # Parent/launcher stopped; never leave an unowned simulator running.
    except BaseException:
        try:
            connection.send({"error": traceback.format_exc()})
        except (EOFError, BrokenPipeError):
            pass
        raise
    finally:
        if env is not None:
            env.close()
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct-instruction-suffix", "--correct_instruction_suffix", type=parse_bool, default=True)
    parser.add_argument("--fd", type=int)
    parser.add_argument("--suite")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--describe", type=Path, help="Write exact official task languages; no env/model is created")
    args = parser.parse_args()
    if args.describe:
        _, _, benchmark = official_modules()
        tasks = []
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            source = benchmark.get_benchmark_dict()[suite]()
            for index in range(source.n_tasks):
                task = source.get_task(index)
                tasks.append({"suite": suite, "task_id": index, "task_name": task.name, "language": task_instruction(task, args.correct_instruction_suffix)})
        args.describe.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    else:
        if args.fd is None or args.suite is None or args.task_id is None:
            parser.error("--fd, --suite and --task-id are required")
        serve(Connection(args.fd), args.suite, args.task_id, args.correct_instruction_suffix)


if __name__ == "__main__":
    main()
