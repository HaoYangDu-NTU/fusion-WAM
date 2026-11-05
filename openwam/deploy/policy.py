"""WAM policy facade: one obs→action entry point over the two executors.

``WAMPolicy`` is the seam between the server (which hands it preprocessed
observations) and the execution mechanism (which schedules engine calls):

- sync mode (default): :class:`SyncInferenceExecutor` — blocking
  buffer-and-replan with a bounded execution horizon.
- async mode: :class:`AsyncInferenceExecutor` — double-buffered background
  inference overlapping generation with execution.

The executor is chosen once at construction from the normalized async
config; per-step dispatch is plain delegation.
"""

from collections import deque

import numpy as np

from openwam.history import condition_frame_offsets, select_condition_images
from openwam.deploy.engine import BaseInferenceEngine
from openwam.deploy.executors import (
    AsyncInferenceExecutor,
    SyncInferenceExecutor,
    normalize_execution_config,
)


class WAMPolicy:
    """Unified policy facade over the sync / async execution mechanisms.

    Args:
        engine: Inference engine that generates action chunks.
        cfg: Root config, retained for policy-level consumers.
        execution_config: ExecutionConfig-like. ``inference_horizon`` applies
            to both modes; ``inference_delay_steps`` applies only to async.
    """

    def __init__(self, engine: BaseInferenceEngine, cfg, execution_config=None):
        self.cfg = cfg
        self.engine = engine
        backbone = getattr(getattr(engine, "architecture", None), "video_backbone", None)
        self._condition_offsets = getattr(getattr(backbone, "video_encoder", None), "clean_condition_frame_offsets", None)
        self._history_num_tubelets = getattr(
            getattr(backbone, "video_encoder", None), "history_num_tubelets", 0
        )
        self._image_history = bool(getattr(
            getattr(backbone, "video_encoder", None), "add_one_history_context", False
        )) and not self._history_num_tubelets
        self._history_images = deque(
            maxlen=(1 - self._condition_offsets[0]) if self._condition_offsets is not None else
            (5 if self._image_history else 1 - condition_frame_offsets(self._history_num_tubelets)[0])
        )

        self._execution_config = normalize_execution_config(execution_config)
        self._async = self._execution_config.enabled
        if self._async:
            self._executor = AsyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
                inference_delay_steps=self._execution_config.inference_delay_steps,
            )
        else:
            self._executor = SyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
            )

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action for the given (already preprocessed) observation.

        The final legality projection for two-point command dims
        (``architecture.binary_command_dims``, from the CKPT's dataloader.binary_action_dims) runs
        HERE — after all executor arithmetic. The normalizer already emits exact ±1 for those dims,
        and this final boundary also protects engines or checkpoints that emit
        values between the two legal commands. Threshold 0.5 preserves the
        downstream command contract for the WS server and direct consumers.
        """
        action = self._executor.predict_action(self._build_conditions(obs))
        dims = getattr(getattr(self.engine, "architecture", None), "binary_command_dims", ()) or ()
        if dims:
            action = np.array(action)
            for d in dims:
                if d >= action.shape[-1]:
                    raise ValueError(
                        f"binary_command_dims includes {d} but the action is {action.shape[-1]}-D; "
                        "the ckpt config and the served action width disagree."
                    )
                action[..., d] = np.where(action[..., d] > 0.5, 1.0, -1.0)
        return action

    def reset(self):
        """Clear executor state between episodes."""
        self._executor.reset()
        self._history_images.clear()

    def shutdown(self):
        """Release executor resources (background threads in async mode)."""
        self._executor.shutdown()

    def _build_conditions(self, obs: dict) -> dict:
        """Assemble inference conditions from the current observation.

        Populates the engine-facing fields (``clean_condition_images``,
        ``prompt``) from the server-preprocessed observation so the
        pipeline receives images without any further client-side work.
        """
        conditions = {
            "observation": obs,
        }
        img = obs.get("image")
        if img is not None:
            if self._condition_offsets is not None:
                self._history_images.append(img)
                conditions["clean_condition_images"] = select_condition_images(self._history_images, frame_offsets=self._condition_offsets)
            elif self._history_num_tubelets:
                # Called at every control step, including steps served from the
                # action buffer. Missing past frames repeat episode frame 0.
                self._history_images.append(img)
                conditions["clean_condition_images"] = select_condition_images(
                    self._history_images, self._history_num_tubelets
                )
            elif self._image_history:
                self._history_images.append(img)
                conditions["clean_condition_images"] = [self._history_images[0], img]
            else:
                conditions["clean_condition_images"] = [img]
        if obs.get("prompt"):
            conditions["prompt"] = obs["prompt"]
        if "state" in obs and obs["state"] is not None:
            conditions["proprio"] = obs["state"]
        return conditions
