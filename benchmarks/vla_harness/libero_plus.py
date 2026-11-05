"""Official OpenWAM LIBERO-plus as an independent harness benchmark."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
from pathlib import Path

from benchmarks.vla_harness.libero import OpenWAMLIBEROBenchmark, make_observation
from benchmarks.vla_harness.libero_plus_protocol import load_config, task_catalog


class OfficialPlusEnvironment:
    """Keep the official fresh-process RNG/import semantics inside each worker."""

    def __init__(self, task, timeout=300, correct_instruction_suffix=True):
        self.timeout = timeout
        self.connection, child = multiprocessing.Pipe()
        try:
            self.process = subprocess.Popen(
                [
                    os.environ["LIBERO_PLUS_PYTHON"],
                    "-m",
                    "benchmarks.vla_harness.libero_plus_env",
                    "--fd",
                    str(child.fileno()),
                    "--suite",
                    task["suite"],
                    "--task-id",
                    str(task["task_id"]),
                    "--correct-instruction-suffix",
                    str(correct_instruction_suffix),
                ],
                pass_fds=(child.fileno(),),
                # Inherit the harness worker's process group, CPU affinity,
                # render device and thread limits; no training process is touched.
            )
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()
        try:
            self.initial = self.receive()
            if self.initial["task_name"] != task["task_name"]:
                raise ValueError("Official LIBERO-plus task order differs from the planned catalog")
        except BaseException:
            self.close()
            raise

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError("Official LIBERO-plus environment did not respond")
        try:
            result = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"Official LIBERO-plus process exited: {self.process.poll()}") from exc
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def step(self, action):
        self.connection.send(("step", action.tolist()))
        return self.receive()["result"]

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.connection.send(("close", None))
                except (BrokenPipeError, OSError):
                    pass
        finally:
            self.connection.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()


class OpenWAMLIBEROPlusBenchmark(OpenWAMLIBEROBenchmark):
    def __init__(self, suites=None, task_ids=None, correct_instruction_suffix=True, excluded_tasks=None, seperate_encode_view=False, policy_backend="openwam", categories=None, evaluation_capabilities=None):
        super().__init__(suites=suites, task_ids=task_ids, seperate_encode_view=seperate_encode_view, policy_backend=policy_backend,
                         evaluation_capabilities=evaluation_capabilities)
        if not isinstance(correct_instruction_suffix, bool):
            raise TypeError("correct_instruction_suffix must be a boolean")
        self.correct_instruction_suffix = correct_instruction_suffix
        self.categories = categories
        self.excluded_tasks = {tuple(pair) for pair in (excluded_tasks or [])}
        self.cfg = load_config()
        self.cfg["policy_backend"] = policy_backend
        self.cfg["seperate_encode_view"] = bool(seperate_encode_view)
        self._language = None

    def get_tasks(self):
        tasks = task_catalog(Path(os.environ["LIBERO_PLUS_PATH"]), self.suites, self.task_ids, self.categories)
        return [t for t in tasks if (t["suite"], t["task_id"]) not in self.excluded_tasks]

    @staticmethod
    def shard_work_items(items, num_shards, shard_id):
        queue = os.environ.get("OPENWAM_DYNAMIC_TASK_QUEUE")
        if queue:
            from benchmarks.vla_harness.task_queue import DynamicWorkItems
            return DynamicWorkItems(queue, items)
        return OpenWAMLIBEROBenchmark.shard_work_items(items, num_shards, shard_id)

    def reset(self, task):
        if task["episode_idx"] != 0:
            raise ValueError("Official LIBERO-plus requires exactly trial 0 for every task")
        self.cleanup()
        self._env = OfficialPlusEnvironment(
            task, timeout=int(self.cfg["request_timeout"]),
            correct_instruction_suffix=self.correct_instruction_suffix,
        )
        self._steps = 0
        self._max_steps = int(self.cfg["max_steps_by_suite"][task["suite"]])
        self._language = self._env.initial["language"]
        return self._env.initial["obs"]

    def make_obs(self, raw_obs, task):
        # The child and cache exporter use the same optional suffix resolver.
        with self.performance.measure("observation_prepare"):
            return make_observation(raw_obs, self._language, self.cfg)

    def step(self, action):
        result = super().step(action)
        # A healthy glass-noise episode can exceed the watchdog's stall limit.
        # Report completed work; never pet a timer while a simulator call hangs.
        from vla_eval import watchdog

        watchdog.pet(f"OpenWAMLIBEROPlus completed step {self._steps}")
        return result

    def get_metadata(self):
        return {
            "correct_instruction_suffix": self.correct_instruction_suffix,
            "categories": self.categories,
            "max_steps": 700,
            "max_episodes_per_task": 1,
            "protocol": "openwam-libero-plus-native-action-seed10000-settle30-v1",
        }
