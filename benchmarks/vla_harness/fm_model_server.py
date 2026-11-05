"""Native FM-VJEPA-Policy behind the existing harness transport/action queue."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
from benchmarks.vla_harness.chunk_server import EvaluationModelServer
from benchmarks.vla_harness.execution import EvaluationCapabilities
from vla_eval.specs import IMAGE_RGB, LANGUAGE, POSITION_DELTA, ROTATION_AA, DimSpec
from benchmarks.vla_harness.fm_policy import FM_ROOT


class FMVJEPAModelServer(EvaluationModelServer):
    def __init__(self, checkpoint_dir: str, checkpoint_name: str, encoder: str, text_cache: str,
                 device: str = 'cuda:0', max_batch_size: int = 8, max_wait_time: float = .01,
                 client_action_queue: bool = False):
        # Native FM predicts 32 actions and executes 16 before replanning.
        super().__init__(chunk_size=16, max_batch_size=max_batch_size, max_wait_time=max_wait_time)
        self._episodes, self._history = {}, {}
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fm-vjepa-gpu')
        try:
            self._thread.submit(self._load, checkpoint_dir, checkpoint_name, encoder, text_cache, device).result()
        except BaseException:
            self._thread.shutdown()
            raise
        self.configure_evaluation(EvaluationCapabilities(
            execution_horizon=16, history_offsets=(-4, 0), history_key="history_images", prediction_horizon=32,
        ), client_action_queue)

    def _load(self, directory, name, encoder, text_cache, device):
        from pathlib import Path
        import torch
        sys.path.insert(0, str(FM_ROOT / 'src'))
        from vjepa_policy.policy_serving import PolicyServingConfig, VJEPAPolicyServing
        np.random.seed(7)
        torch.manual_seed(7)
        checkpoint = str(Path(directory) / name)
        cfg = PolicyServingConfig.from_checkpoint(checkpoint, overrides={'dataset_stats': str(Path(directory) / 'dataset_stats.json')})
        if not cfg.flow_matching or cfg.action_dim != 7 or cfg.proprio_dim != 8 or cfg.t5_len != 128:
            raise ValueError('Expected native FM LIBERO checkpoint (action7/state8/T5-128)')
        self.policy = VJEPAPolicyServing(checkpoint, encoder, text_cache, config=cfg, device=device)

    async def on_episode_start(self, config, ctx):
        await super().on_episode_start(config, ctx)
        self._episodes[ctx.session_id] = ctx.episode_id
        self._history.pop(ctx.session_id, None)

    async def on_episode_end(self, result, ctx):
        await super().on_episode_end(result, ctx)
        if self._episodes.get(ctx.session_id) == ctx.episode_id:
            self._episodes.pop(ctx.session_id, None)
            self._history.pop(ctx.session_id, None)

    async def on_observation(self, obs, ctx):
        if getattr(self, "client_action_queue", False):
            return await super().on_observation(obs, ctx)
        if self._episodes.get(ctx.session_id) != ctx.episode_id:
            raise RuntimeError('Observation is outside the active episode')
        # Runs on EVERY environment step, including the 15 buffered-action steps.
        history = self._history.setdefault(ctx.session_id, deque(maxlen=5))
        history.append(obs['images'])
        await super().on_observation({**obs, 'history_images': history[0]}, ctx)

    @staticmethod
    def native_observation(obs):
        import torch
        state = np.asarray(obs['state'], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('Expected finite native 8D proprioception')
        result = {'observation.state': torch.from_numpy(state).unsqueeze(0), 'task': [obs['prompt']]}
        for slot, key in [('head_camera', 'observation.images.image'), ('left_wrist_camera', 'observation.images.wrist_image')]:
            for images, suffix in [(obs['images'], ''), (obs['history_images'], '_past')]:
                image = np.asarray(images[slot])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError('Expected unresized uint8 RGB camera image')
                result[key + suffix] = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.
        return result

    def predict_batch(self, obs_batch, ctx_batch):
        for ctx in ctx_batch:
            if self._episodes.get(ctx.session_id) != ctx.episode_id:
                raise RuntimeError('Prediction is outside the active episode')
        with self.performance.measure("batch_dispatch_and_predict", len(obs_batch)):
            self.performance.add(f"batch_size_{len(obs_batch)}")
            return self._thread.submit(self._predict, obs_batch).result()

    def _predict(self, observations):
        outputs = self.policy.infer_batch([self.native_observation(obs) for obs in observations])
        for output in outputs:
            actions = np.asarray(output['actions'])
            if actions.shape != (32, 7) or not np.isfinite(actions).all():
                raise ValueError('Native FM returned invalid 32x7 actions')
        # Native serving already unnormalizes and converts the gripper once.
        return outputs

    def get_action_spec(self):
        return {'position': POSITION_DELTA, 'rotation': ROTATION_AA,
                'gripper': DimSpec('gripper', 1, 'continuous_close_positive', (-1, 1))}

    def get_observation_spec(self):
        return {'head_camera': IMAGE_RGB, 'left_wrist_camera': IMAGE_RGB, 'language': LANGUAGE,
                'state': DimSpec('state', 8, 'eef_pos3_axisangle3_gripper_qpos2')}


if __name__ == '__main__':
    from vla_eval.model_servers.serve import run_server
    run_server(FMVJEPAModelServer)
