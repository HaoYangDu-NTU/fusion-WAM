"""OpenWAM's native LIBERO protocol implemented as a harness StepBenchmark."""

from __future__ import annotations

import sys
from itertools import groupby
from pathlib import Path

import numpy as np
import yaml
from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.orchestrator import _shard_work_items
from vla_eval.render import configure_mujoco_render
from vla_eval.specs import IMAGE_RGB, LANGUAGE, POSITION_DELTA, ROTATION_AA, DimSpec

from benchmarks.utils import libero_obs_to_eef10, prepare_libero_camera
from benchmarks.vla_harness.execution import PhaseStats

# single_eval uses a script-local import; load its helpers without editing it.
_OFFICIAL = Path(__file__).resolve().parents[1] / "libero"
if str(_OFFICIAL) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL))
from benchmarks.libero import single_eval  # noqa: E402

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STATE_SPEC = DimSpec("state", 10, "achieved_eef_pos3_rot6d_gripper_open_scale1")
GRIPPER_SPEC = DimSpec("gripper", 1, "continuous_close_positive", (-1, 1))
ACTION_SPEC = {"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_SPEC}
OBSERVATION_SPEC = {
    "head_camera": IMAGE_RGB,
    "left_wrist_camera": IMAGE_RGB,
    "language": LANGUAGE,
    "state": STATE_SPEC,
}


def make_observation(obs: dict, prompt: str, cfg: dict) -> dict:
    """Same pixels/state as OpenWAMLiberoPolicy.act; harness handles lossless transport."""
    fm = cfg.get("policy_backend", "openwam") in ("fm-vjepa", "vpp2")
    images = {}
    for slot, key in (("head_camera", cfg["head_camera_key"]), ("left_wrist_camera", cfg["left_wrist_camera_key"])):
        image = np.asarray(obs[key]).astype(np.uint8, copy=False)
        if cfg["image_transform"] == "rotate_180":
            image = image[::-1, ::-1]
        images[slot] = prepare_libero_camera(image, slot, seperate_encode_view=fm or cfg.get("seperate_encode_view", False))
    images["right_wrist_camera"] = None
    if fm:
        from benchmarks.vla_harness.fm_policy import native_state
        state = native_state(obs)
    else:
        state = libero_obs_to_eef10(obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"])
    return {
        "images": images,
        "prompt": prompt,
        "state": state,
    }


class OpenWAMLIBEROBenchmark(StepBenchmark):
    render_backends = frozenset({"gpu", "cpu"})
    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success", "action"})

    @classmethod
    def configure_render(cls, mode):
        return configure_mujoco_render(mode)

    def __init__(self, suites: list[str] | None = None, task_ids: list[int] | None = None, seperate_encode_view=False,
                 excluded_tasks=None, policy_backend="openwam", evaluation_capabilities=None):
        super().__init__()
        self.cfg = yaml.safe_load((_OFFICIAL / "policy_config.yml").read_text())
        if policy_backend not in ("openwam", "fm-vjepa", "vpp2"):
            raise ValueError("Unknown policy backend")
        self.cfg["policy_backend"] = policy_backend
        self.evaluation_capabilities = evaluation_capabilities
        self.performance = PhaseStats("environment")
        self.cfg["seperate_encode_view"] = bool(seperate_encode_view)
        self.suites = list(SUITES if suites is None else suites)
        if not self.suites or len(set(self.suites)) != len(self.suites) or any(s not in SUITES for s in self.suites):
            raise ValueError(f"suites must be distinct entries from {SUITES}")
        self.task_ids = task_ids
        self.excluded_tasks = {tuple(pair) for pair in (excluded_tasks or [])}
        self._env = None
        self._current_task = None
        self._next_trial = 0
        self._steps = 0
        self._max_steps = 0

    @staticmethod
    def shard_work_items(items, num_shards, shard_id):
        """Use harness shuffling on whole tasks, preserving each task's environment RNG."""
        groups = [(idx, list(group)) for idx, group in groupby(items, key=lambda item: item[0])]
        return [item for _, group in _shard_work_items(groups, num_shards, shard_id) for item in group]

    def get_tasks(self):
        single_eval._write_libero_config()
        from libero.libero import benchmark

        tasks = []
        for suite in self.suites:
            source = benchmark.get_benchmark_dict()[suite]()
            ids = range(source.n_tasks) if self.task_ids is None else self.task_ids
            if len(set(ids)) != len(ids) or any(i < 0 or i >= source.n_tasks for i in ids):
                raise ValueError("task_ids must be distinct valid task indices")
            for index in ids:
                task = source.get_task(index)
                tasks.append(
                    {
                        "name": f"{suite}/{index:02d}/{task.name}",
                        "suite": suite,
                        "task_id": index,
                        "language": task.language,
                        "task_obj": task,
                        "init_states": source.get_task_init_states(index),
                    }
                )
        return [t for t in tasks if (t["suite"], t["task_id"]) not in self.excluded_tasks]

    def reset(self, task):
        key = (task["suite"], task["task_id"])
        trial = task["episode_idx"]
        expected = self._next_trial if key == self._current_task else 0
        if trial != expected:
            raise ValueError("OpenWAM trials must be consecutive; use the bundled task-sharding hook")
        if self._current_task != key:
            self.cleanup()
            self._env = single_eval._make_env_with_randomization_retries(task["task_obj"], self.cfg)
            self._env.seed(int(self.cfg["seed"]))
            self._current_task = key
        self._next_trial = trial + 1
        self._steps = 0
        self._max_steps = single_eval._resolve_max_steps(self.cfg, task["suite"])
        obs = self._env.reset()
        states = task["init_states"]
        if len(states):
            obs = self._env.set_init_state(states[trial % len(states)])
        settle = single_eval._parse_settle_action(self.cfg["settle_action"])
        for _ in range(self.cfg["settle_steps"]):
            obs, _, _, _ = self._env.step(settle)
        return obs

    def make_obs(self, raw_obs, task):
        with self.performance.measure("observation_prepare"):
            return make_observation(raw_obs, task["language"], self.cfg)

    async def start_episode(self, task, recorder=None):
        with self.performance.measure("episode_initialize"):
            await super().start_episode(task, recorder=recorder)

    def step(self, action):
        value = np.asarray(action["actions"], dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("Expected a finite LIBERO action with shape (7,)")
        with self.performance.measure("environment_step"):
            obs, reward, done, info = self._env.step(value)
        self._steps += 1
        self._recorder.record_video(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        self._recorder.record_step(reward=float(reward), done=bool(done), success=bool(done), action=value)
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def check_done(self, result):
        return bool(result.done) or self._steps >= self._max_steps

    def get_step_result(self, result):
        return {"success": bool(result.done)}

    def get_metadata(self):
        return {"max_steps": 700, "max_episodes_per_task": 50, "protocol": "openwam-native-task-rng-v1"}

    def get_action_spec(self):
        return dict(ACTION_SPEC)

    def get_observation_spec(self):
        spec = dict(OBSERVATION_SPEC)
        if self.cfg.get("policy_backend") in ("fm-vjepa", "vpp2"):
            spec["state"] = DimSpec("state", 8, "eef_pos3_axisangle3_gripper_qpos2")
        return spec

    def cleanup(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self.performance.emit(control_steps=self._steps)
        self._current_task = None
