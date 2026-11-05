"""Run a contiguous episode batch for one task in a LIBERO suite.

This keeps each MuJoCo client process short-lived while the policy server is
kept alive by the shell evaluator.  It delegates the rollout itself to the
repository's canonical LIBERO client.
"""

from __future__ import annotations

import os

from examples.libero import run_libero_client as client


def _fixed_suite_factory(original_factory, task_id: int, episode_offset: int):
    base_suite = original_factory()

    class FixedSuite:
        n_tasks = 1

        def get_task(self, _local_id: int):
            return base_suite.get_task(task_id)

        def get_task_init_states(self, _local_id: int):
            return base_suite.get_task_init_states(task_id)[episode_offset:]

    return FixedSuite()


def main() -> None:
    task_id = int(os.environ["LIBERO_FIXED_TASK_ID"])
    episode_offset = int(os.environ.get("LIBERO_EPISODE_OFFSET", "0"))
    suite_name = os.environ["LIBERO_FIXED_SUITE"]
    original_get_dict = client.benchmark.get_benchmark_dict

    def fixed_get_dict():
        original = original_get_dict()
        fixed = dict(original)
        fixed[suite_name] = lambda: _fixed_suite_factory(
            original[suite_name], task_id, episode_offset
        )
        return fixed

    client.benchmark.get_benchmark_dict = fixed_get_dict
    # Videos are not part of the evaluation metric and materially slow the
    # many short client processes used to isolate MuJoCo renderer lifetime.
    client.imageio.mimwrite = lambda *args, **kwargs: None
    client.tyro.cli(client.eval_libero)


if __name__ == "__main__":
    main()
