"""Batched VPP2 with the harness action queue and bounded GPU memory."""
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import gc
from collections import Counter
from benchmarks.vla_harness.chunk_server import EvaluationModelServer
from benchmarks.vla_harness.execution import EvaluationCapabilities
from vla_eval.specs import IMAGE_RGB, LANGUAGE, POSITION_DELTA, ROTATION_AA, DimSpec
from benchmarks.vla_harness.vpp2_policy import VPP2Policy


class VPP2ModelServer(EvaluationModelServer):
    def __init__(self, checkpoint_dir: str, device: str = "cuda:0", max_batch_size: int = 32,
                 max_wait_time: float = .01, client_action_queue: bool = False):
        if max_batch_size < 1:
            raise ValueError("Batch size must be positive")
        super().__init__(chunk_size=10, max_batch_size=max_batch_size, max_wait_time=max_wait_time)
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vpp2-gpu")
        try:
            self.policy = self._thread.submit(VPP2Policy, checkpoint_dir, device).result()
        except BaseException:
            self._thread.shutdown()
            raise
        self._requests = 0
        # Real H100 acceptance: B16 peaks at 73.5 GiB; B32 exceeds 80 GiB.
        self._micro_batch_size = min(max_batch_size, 16)
        self._batch_counts = Counter()
        self._window_started = time.monotonic()
        self._generation_seconds = 0.
        self.configure_evaluation(EvaluationCapabilities(execution_horizon=10, prediction_horizon=32), client_action_queue)

    def predict_batch(self, obs_batch, ctx_batch):
        with self.performance.measure("batch_dispatch_and_predict", len(obs_batch)):
            return self._thread.submit(self._predict, obs_batch).result()

    def _predict(self, observations):
        import torch
        started = time.monotonic()
        results = []
        while len(results) < len(observations):
            offset = len(results)
            samples = observations[offset:offset+self._micro_batch_size]
            try:
                outputs = self.policy.predict_batch(samples)
            except torch.cuda.OutOfMemoryError:
                self.performance.add("oom_retries")
                if len(samples) == 1:
                    raise
                self._micro_batch_size = max(1, len(samples)//2)
                logging.getLogger(__name__).warning("Reducing VPP2 GPU microbatch to %d after OOM", self._micro_batch_size)
            else:
                results.extend(outputs)
                self._batch_counts[len(samples)] += 1
                self.performance.add(f"batch_size_{len(samples)}")
                continue
            gc.collect()
            torch.cuda.empty_cache()
        self._requests += len(results)
        self._generation_seconds += time.monotonic()-started
        elapsed = time.monotonic()-self._window_started
        if elapsed >= 30:
            logging.getLogger(__name__).info("VPP2 window=%.2fs batches=%s requests=%d generation=%.3fs microbatch_limit=%d",
                elapsed, dict(self._batch_counts), sum(n*c for n,c in self._batch_counts.items()),
                self._generation_seconds, self._micro_batch_size)
            self._batch_counts.clear()
            self._generation_seconds = 0.
            self._window_started = time.monotonic()
        return results

    def get_action_spec(self):
        return {"position": POSITION_DELTA, "rotation": ROTATION_AA,
                "gripper": DimSpec("gripper", 1, "continuous_close_positive", (-1, 1))}

    def get_observation_spec(self):
        return {"head_camera": IMAGE_RGB, "left_wrist_camera": IMAGE_RGB, "language": LANGUAGE,
                "state": DimSpec("state", 8, "eef_pos3_axisangle3_gripper_qpos2")}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server
    run_server(VPP2ModelServer)
