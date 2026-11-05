"""Worker-local action consumption; the harness still counts every env step."""

from collections import deque
import uuid

import numpy as np
from vla_eval.connection import Connection

from benchmarks.vla_harness.execution import EvaluationCapabilities, PhaseStats
from openwam.history import select_condition_images


class EvaluationConnection(Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capabilities = None
        self._active_episode = None
        self._control_step = 0
        self._actions = deque()
        self._history = deque(maxlen=1)
        self.performance = PhaseStats("client")

    async def connect(self, **kwargs):
        await super().connect(**kwargs)
        advertised = self.server_info.get("observation_params", {}).get("evaluation_capabilities")
        if advertised:
            self._capabilities = EvaluationCapabilities.from_dict(advertised["model"])
            self._local_chunks = advertised["client_action_queue"]
            if type(self._local_chunks) is not bool:
                raise ValueError("client_action_queue must be boolean")
            if self._local_chunks and self._capabilities.requires_step_observe:
                raise ValueError("Cannot skip observations required by this model")
            self._history = deque(maxlen=1 - self._capabilities.history_offsets[0])
        else:
            self._local_chunks = False

    async def start_episode(self, config):
        self._actions.clear()
        self._history.clear()
        self._control_step = 0
        self._active_episode = None
        await super().start_episode(config)
        self._active_episode = uuid.uuid4().hex

    async def act(self, obs):
        if not getattr(self, "_local_chunks", False):
            with self.performance.measure("request_roundtrip"):
                response = await super().act(obs)
            self.performance.add("control_actions")
            return response
        if self._active_episode is None:
            raise RuntimeError("Action requested outside an active episode")
        # Keep control-step history even when no inference request is sent.
        if self._capabilities.history_key:
            # Simulators may recycle their image buffers; the old wire path
            # made independent copies on decode, so local history must too.
            self._history.append({key: None if image is None else np.array(image, copy=True)
                                  for key, image in obs["images"].items()})
        if not self._actions:
            request = {**obs, "_execution": {"episode": self._active_episode, "step": self._control_step}}
            if self._capabilities.history_key:
                frames = select_condition_images(self._history, frame_offsets=self._capabilities.history_offsets)
                request[self._capabilities.history_key] = frames[0] if self._capabilities.history_key == "history_images" else frames
            # Automatic reconnect would lose server episode/KV state. Fail the
            # episode instead of replaying a request into a fresh session.
            if not self.is_connected:
                raise ConnectionError("Evaluation connection lost during an episode")
            with self.performance.measure("request_roundtrip"):
                response = await super().act(request)
            if response.get("_execution") != request["_execution"]:
                raise RuntimeError("Stale or misordered action chunk")
            actions = np.asarray(response["actions"], dtype=np.float32)
            if (actions.ndim != 2 or not 0 < len(actions) <= self._capabilities.execution_horizon
                    or not np.isfinite(actions).all()):
                raise ValueError("Invalid executable action chunk")
            self._actions.extend(action.copy() for action in actions)
        action = self._actions.popleft()
        self._control_step += 1
        self.performance.add("control_actions")
        return {"actions": action}

    async def _ensure_connected(self):
        if (self._active_episode is not None and getattr(self, "_local_chunks", False)
                and not self.is_connected):
            raise ConnectionError("Evaluation connection lost during an episode")
        await super()._ensure_connected()

    async def end_episode(self, result):
        try:
            await super().end_episode(result)
        finally:
            self.performance.emit(episode=self._active_episode)
            self._active_episode = None
            self._actions.clear()
            self._history.clear()

    async def close(self):
        self._active_episode = None
        self._actions.clear()
        self._history.clear()
        await super().close()


def install_evaluation_connection():
    """Select the local client in this worker only; never edit installed harness."""
    from vla_eval import orchestrator

    orchestrator.Connection = EvaluationConnection
