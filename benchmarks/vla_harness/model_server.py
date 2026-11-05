"""OpenWAM model integration for harness session buffering and batched dispatch."""

from __future__ import annotations

import ctypes
import gc
import logging
import time
from pathlib import Path
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
from benchmarks.vla_harness.batch_engine import BatchedOpenWAMEngine
from benchmarks.vla_harness.chunk_server import EvaluationModelServer
from benchmarks.vla_harness.execution import EvaluationCapabilities
from benchmarks.vla_harness.libero import ACTION_SPEC, OBSERVATION_SPEC
from openwam.history import condition_frame_offsets, select_condition_images


def load_native_server(checkpoint_dir, checkpoint_name, device, denoise_steps, inference_horizon, compile_enabled):
    from omegaconf import OmegaConf

    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

    checkpoint = Path(checkpoint_dir).resolve()
    deploy_config = next((parent / "vla_harness_deploy.yaml"
                          for parent in (checkpoint, *checkpoint.parents)
                          if (parent / "vla_harness_deploy.yaml").is_file()), None)
    cfg = _load_deploy_yaml(str(deploy_config)) if deploy_config else _load_deploy_yaml()
    logging.getLogger(__name__).info(
        "Evaluation deploy config=%s; dit_cache.enabled=%s",
        deploy_config or "configs/deploy.yaml", cfg.optimization.dit_cache.enabled,
    )
    overrides = {
        "inference.denoise_steps": denoise_steps,
        "inference.denoise_mode": "sync",
        "inference.inference_mode": "sync",
        "inference.inference_horizon": inference_horizon,
        "optimization.compile.enabled": compile_enabled,
        "optimization.decode_video": False,
        "optimization.dit_cache.enabled": False,
    }
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    server = build_server_from_config(cfg, checkpoint_dir, device=device, ckpt_name=checkpoint_name)
    server._init_policy()
    return server


def preprocess_observation(preprocessor, obs):
    # Official preprocessing accepts PIL/bytes/base64. Harness decodes to arrays;
    # this boundary replaces transport encoding only, without changing pixels.
    payload = {
        **obs,
        "images": {
            key: Image.fromarray(value) if isinstance(value, np.ndarray) else value
            for key, value in obs["images"].items()
        },
    }
    result = preprocessor.preprocess(payload)
    if "history_images" in obs:
        result["history_image"] = preprocessor.preprocess({
            **payload, "images": {
                key: Image.fromarray(value) if isinstance(value, np.ndarray) else value
                for key, value in obs["history_images"].items()
            },
        })["image"]
    if "clean_condition_images" in obs:
        result["clean_condition_images"] = [
            preprocessor.preprocess({
                **payload,
                "images": {
                    key: Image.fromarray(value) if isinstance(value, np.ndarray) else value
                    for key, value in images.items()
                },
            })["image"]
            for images in obs["clean_condition_images"]
        ]
    return result


class OpenWAMModelServer(EvaluationModelServer):
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_name: str = "checkpoint_step_10690.safetensors",
        device: str = "cuda:0",
        denoise_steps: int = 10,
        inference_horizon: int = 10,
        compile_enabled: bool = False,
        max_batch_size: int = 8,
        max_wait_time: float = 0.01,
        client_action_queue: bool = False,
    ):
        if max_batch_size < 1 or max_wait_time < 0 or inference_horizon < 1:
            raise ValueError("Invalid batch/horizon settings")
        # Harness owns the ONLY executable action queue. The native WAMPolicy
        # remains available as a reference but is never called on this path.
        super().__init__(chunk_size=inference_horizon, max_batch_size=max_batch_size, max_wait_time=max_wait_time)
        self._episodes = {}
        self._history_images = {}
        self._batch_counts = Counter()
        self._batch_prepare_seconds = self._batch_inference_seconds = 0.0
        # AnyIO limits concurrent calls but shares a pool with transport work;
        # successive batches can run on different threads. Keep CUDA model
        # construction and inference on one owner thread for the whole service.
        self._inference_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openwam-gpu")
        try:
            self._inference_thread.submit(
                self._load_model,
                checkpoint_dir,
                checkpoint_name,
                device,
                denoise_steps,
                inference_horizon,
                compile_enabled,
            ).result()
        except BaseException:
            self._inference_thread.shutdown()
            raise
        offsets = self._condition_offsets
        if offsets is not None or self._history_num_tubelets:
            offsets = tuple(offsets) if offsets is not None else tuple(condition_frame_offsets(self._history_num_tubelets))
            key = "clean_condition_images"
        elif self._image_history:
            offsets, key = (-4, 0), "history_images"
        else:
            offsets, key = (0,), None
        self.configure_evaluation(EvaluationCapabilities(
            execution_horizon=inference_horizon, history_offsets=offsets, history_key=key,
        ), client_action_queue)
        self._batch_window_started = time.monotonic()

    def _load_model(self, checkpoint_dir, checkpoint_name, device, denoise_steps, inference_horizon, compile_enabled):
        native = load_native_server(
            checkpoint_dir, checkpoint_name, device, denoise_steps, inference_horizon, compile_enabled
        )
        if native._ckpt_contract().get("representation") != "eef":
            raise ValueError("OpenWAM LIBERO requires the native EEF checkpoint contract")
        self.native_server = native
        self.engine = BatchedOpenWAMEngine(native.engine)
        self.preprocessor = native._obs_preprocessor
        backbone = getattr(getattr(native.engine, "architecture", None), "video_backbone", None)
        self._condition_offsets = getattr(getattr(backbone, "video_encoder", None), "clean_condition_frame_offsets", None)
        self._history_num_tubelets = getattr(
            getattr(backbone, "video_encoder", None), "history_num_tubelets", 0
        )
        self._image_history = bool(getattr(
            getattr(backbone, "video_encoder", None), "add_one_history_context", False
        )) and not self._history_num_tubelets
        # The native loader has already moved model weights to CUDA. Return
        # freed CPU construction/checkpoint allocations to the OS (glibc).
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)

    async def on_episode_start(self, config, ctx):
        await super().on_episode_start(config, ctx)
        self._episodes[ctx.session_id] = ctx.episode_id
        self._history_images.pop(ctx.session_id, None)

    async def on_episode_end(self, result, ctx):
        await super().on_episode_end(result, ctx)
        if self._episodes.get(ctx.session_id) == ctx.episode_id:
            self._episodes.pop(ctx.session_id)
            self._history_images.pop(ctx.session_id, None)

    async def on_observation(self, obs, ctx):
        if getattr(self, "client_action_queue", False):
            return await super().on_observation(obs, ctx)
        offsets = getattr(self, "_condition_offsets", None)
        if offsets is not None or self._history_num_tubelets or self._image_history:
            if self._episodes.get(ctx.session_id) != ctx.episode_id:
                raise RuntimeError("Observation is outside the active session episode")
            # Record every control step before the harness serves buffered actions.
            history = self._history_images.setdefault(
                ctx.session_id, deque(
                    maxlen=(1 - offsets[0]) if offsets is not None else
                    (5 if self._image_history else 1 - condition_frame_offsets(self._history_num_tubelets)[0])
                )
            )
            history.append(obs["images"])
            if offsets is not None:
                obs = {**obs, "clean_condition_images": select_condition_images(history, frame_offsets=offsets)}
            elif self._image_history:
                obs = {**obs, "history_images": history[0]}
            else:
                obs = {**obs, "clean_condition_images": select_condition_images(history, self._history_num_tubelets)}
        await super().on_observation(obs, ctx)

    def predict_batch(self, obs_batch, ctx_batch):
        with self.performance.measure("batch_dispatch_and_predict"):
            return self._inference_thread.submit(self._predict_batch, obs_batch, ctx_batch).result()

    def _predict_batch(self, obs_batch, ctx_batch):
        for obs, ctx in zip(obs_batch, ctx_batch):
            if self._episodes.get(ctx.session_id) != ctx.episode_id:
                raise RuntimeError("Observation is outside the active session episode")
            state = np.asarray(obs["state"])
            if state.shape != (10,) or not np.isfinite(state).all():
                raise ValueError("Expected finite achieved EEF10 proprioception")
        started = time.monotonic()
        observations = [preprocess_observation(self.preprocessor, obs) for obs in obs_batch]
        prepared = time.monotonic()
        chunks = self.engine.generate_batch(observations)
        finished = time.monotonic()
        self.performance.add("preprocess", prepared - started, len(observations))
        self.performance.add("model_predict", finished - prepared, len(observations))
        self.performance.add(f"batch_size_{len(observations)}")
        self._batch_counts[len(obs_batch)] += 1
        self._batch_prepare_seconds += prepared - started
        self._batch_inference_seconds += finished - prepared
        elapsed = finished - self._batch_window_started
        if elapsed >= 30:
            logging.getLogger(__name__).info(
                "Inference window=%.1fs batches=%s requests=%d preprocessing=%.3fs generation=%.3fs",
                elapsed, dict(sorted(self._batch_counts.items())),
                sum(size * count for size, count in self._batch_counts.items()),
                self._batch_prepare_seconds, self._batch_inference_seconds,
            )
            self._batch_counts.clear()
            self._batch_prepare_seconds = self._batch_inference_seconds = 0.0
            self._batch_window_started = finished
        results = []
        for chunk in chunks:
            if len(chunk) < self.chunk_size:
                raise ValueError("inference_horizon exceeds the generated action chunk")
            actions = np.stack([native_eef10_to_libero7d(action) for action in chunk[: self.chunk_size]])
            results.append({"actions": actions})
        return results

    def get_action_spec(self):
        return dict(ACTION_SPEC)

    def get_observation_spec(self):
        return dict(OBSERVATION_SPEC)


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(OpenWAMModelServer)
