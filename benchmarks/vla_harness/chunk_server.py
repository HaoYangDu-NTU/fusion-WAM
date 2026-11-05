"""Shared optional full-chunk transport over the existing batch dispatcher."""

import numpy as np
from vla_eval.model_servers.predict import PredictModelServer

from benchmarks.vla_harness.execution import PhaseStats


class EvaluationModelServer(PredictModelServer):
    async def on_serve_start(self, tg):
        self._execution_sessions.clear()
        await super().on_serve_start(tg)

    def configure_evaluation(self, capabilities, client_action_queue):
        if type(client_action_queue) is not bool:
            raise ValueError("client_action_queue must be boolean")
        if client_action_queue and capabilities.requires_step_observe:
            raise ValueError("This model requires every intermediate observation")
        if self.max_batch_size > 1 and not capabilities.native_batch:
            raise ValueError("Model does not support native batching")
        self.evaluation_capabilities = capabilities
        self.client_action_queue = client_action_queue
        self._execution_sessions = {}
        self.performance = PhaseStats("model")

    def get_observation_params(self):
        return {"evaluation_capabilities": {
            "model": self.evaluation_capabilities.to_dict(),
            "client_action_queue": self.client_action_queue,
        }}

    async def on_episode_start(self, config, ctx):
        if getattr(self, "client_action_queue", False):
            # None makes PredictModelServer return a whole array, while its
            # native dispatcher still merges simultaneous inference requests.
            self._session_chunk_sizes[ctx.session_id] = None
            self._execution_sessions[ctx.session_id] = {
                "episode_id": ctx.episode_id, "token": None, "step": 0, "pending": False,
            }
        await super().on_episode_start(config, ctx)

    async def on_observation(self, obs, ctx):
        if getattr(self, "client_action_queue", False):
            state = self._execution_sessions.get(ctx.session_id)
            meta = obs.get("_execution", {})
            if (state is None or state["episode_id"] != ctx.episode_id or state["pending"]
                    or type(meta.get("step")) is not int or meta["step"] != state["step"]
                    or not isinstance(meta.get("episode"), str) or not meta["episode"]
                    or state["token"] not in (None, meta["episode"])):
                raise RuntimeError("Invalid episode or control step for chunk request")
            cap = self.evaluation_capabilities
            if cap.history_key:
                frames = obs.get(cap.history_key)
                if frames is None or (cap.history_key == "clean_condition_images"
                                      and len(frames) != len(cap.history_offsets)):
                    raise ValueError("Missing or incorrect control-step history")
            state["token"] = meta["episode"]
            state["pending"] = True
        if hasattr(self, "performance"):
            with self.performance.measure("observation_service"):
                await super().on_observation(obs, ctx)
        else:
            await super().on_observation(obs, ctx)

    async def _process_and_send(self, result, ctx):
        if not getattr(self, "client_action_queue", False):
            return await super()._process_and_send(result, ctx)
        state = self._execution_sessions.get(ctx.session_id)
        if state is None or state["episode_id"] != ctx.episode_id or not state["pending"]:
            raise RuntimeError("Prediction completed outside its active episode")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        horizon = self.evaluation_capabilities.execution_horizon
        if actions.ndim != 2 or len(actions) < horizon or not np.isfinite(actions).all():
            raise ValueError("Model returned an invalid or short action chunk")
        meta = {"episode": state["token"], "step": state["step"]}
        await ctx.send_action({**result, "actions": actions[:horizon], "_execution": meta})
        state["step"] += horizon
        state["pending"] = False
        self.performance.add("executable_actions", count=horizon)

    async def on_episode_end(self, result, ctx):
        state = getattr(self, "_execution_sessions", {}).get(ctx.session_id)
        if state and state["episode_id"] == ctx.episode_id:
            self._execution_sessions.pop(ctx.session_id)
        if hasattr(self, "performance"):
            self.performance.emit(episode=ctx.episode_id)
        await super().on_episode_end(result, ctx)
