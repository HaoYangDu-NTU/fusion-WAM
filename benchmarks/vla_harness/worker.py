"""Run the original harness CLI after limiting this local worker's CPU threads."""

import os


def main():
    # Equivalent to harness's container CPU allocation, using the current conda
    # environment instead of replacing it with a benchmark Docker image.
    shard = int(os.environ["OPENWAM_SHARD_ID"])
    workers = int(os.environ["OPENWAM_NUM_WORKERS"])
    cpus = sorted(os.sched_getaffinity(0))
    width = max(1, len(cpus) // workers)
    selected = cpus[shard * width : (shard + 1) * width] or [cpus[shard % len(cpus)]]
    os.sched_setaffinity(0, selected)
    from vla_eval.protocol.numpy_codec import set_image_format

    # Local loopback has ample bandwidth; preserve pixels without PNG work.
    set_image_format(os.environ.get("OPENWAM_HARNESS_IMAGE_FORMAT", "raw"))
    from benchmarks.vla_harness.client import install_evaluation_connection

    install_evaluation_connection()
    from vla_eval.cli.main import main as harness_main

    harness_main()


if __name__ == "__main__":
    main()
