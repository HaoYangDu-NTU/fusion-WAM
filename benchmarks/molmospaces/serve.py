"""Serve a custom stateless DROID policy on the OpenPI websocket protocol."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from .droid import SCHEMA, normalize
from .evaluate import OPENPI, PROTOCOL, git_head
import sys


class Adapter:
    def __init__(self, policy, stats, chunk):
        self.policy = policy
        self.stats = stats
        self.chunk = chunk

    def infer(self, obs):
        joints = np.asarray(obs["observation/joint_position"], dtype=np.float32)
        grip = np.asarray(obs["observation/gripper_position"], dtype=np.float32).reshape(-1)
        if joints.shape != (7,) or grip.shape != (1,):
            raise ValueError("Invalid DROID robot state")
        state = np.concatenate([joints, grip])
        sample = {
            "images": {
                "exterior_image_1_left": obs["observation/exterior_image_1_left"],
                "wrist_image_left": obs["observation/wrist_image_left"],
            },
            "state": normalize(state, self.stats, "state"),
            "language": obs["prompt"],
        }
        actions = np.asarray(self.policy.predict(sample), dtype=np.float32)
        if actions.shape != (self.chunk, 8) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite normalized actions [{self.chunk},8]")
        return {"actions": normalize(actions, self.stats, "actions", inverse=True)}


def main():
    if git_head(OPENPI) != PROTOCOL["openpi_commit"]:
        raise RuntimeError("OpenPI transport source revision mismatch")
    sys.path.insert(0, str(OPENPI / "src"))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True, help="module:factory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.chunk < 8:
        parser.error("The official client executes eight actions per request")
    module, factory = args.factory.split(":", 1)
    policy = getattr(importlib.import_module(module), factory)(args.checkpoint)
    adapter = Adapter(policy, json.loads(args.stats.read_text()), args.chunk)
    WebsocketPolicyServer(adapter, host="127.0.0.1", port=args.port,
                          metadata={"schema": SCHEMA}).serve_forever()


if __name__ == "__main__":
    main()
