"""Connect the official evaluator to an OpenWAM M29 policy server."""
import os

import numpy as np

from ..openwam2robocasa365_interface import OpenWAMRoboCasa365Policy
from .schema import ACTION_FIELDS, STATE_FIELDS


class M29Policy(OpenWAMRoboCasa365Policy):
    def predict(self, observation):
        state = observation["state"][-1]
        native_observation = {
            f"state.{name}": state[start:end] for name, start, end in STATE_FIELDS
        }
        native_observation.update({
            f"video.{name}": images[-1] for name, images in observation["rgb"].items()
        })
        action = self.act(native_observation, observation["language"])
        flat = np.concatenate([action[f"action.{name}"] for name, _, _ in ACTION_FIELDS])
        if not np.isfinite(flat).all():
            raise ValueError("OpenWAM returned a nonfinite action")
        # Match the native PandaOmron normalized actuator bounds.
        return np.clip(flat[None], -1.0, 1.0)


def make_policy():
    return M29Policy(
        port=int(os.environ["ROBOCASA365_POLICY_PORT"]),
        **({"right_wrist_camera_key": None} if os.environ.get("ROBOCASA365_EVAL_VIEW_COUNT", "2") == "2" else {}),
        seperate_encode_view=True,
    )
