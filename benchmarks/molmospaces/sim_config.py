"""Remote DROID evaluation with the upstream controller and timing defaults."""
import os

from molmo_spaces.configs.policy_configs_baselines import PiPolicyConfig
from molmo_spaces.evaluation.configs.evaluation_configs import PiPolicyEvalConfig
from molmo_spaces.policy.learned_policy.pi_policy import PI_Policy
from molmo_spaces.utils.function_utils import make_lenient


class RemoteDroidPolicy(PI_Policy):
    def obs_to_model_input(self, obs):
        cameras = self.config.policy_config.camera_names
        if not cameras:
            return super().obs_to_model_input(obs)
        # Native PI_Policy reads these fixed keys even when --camera_names is set.
        frame = obs[0] if isinstance(obs, list) else obs
        external, wrist = cameras
        return super().obs_to_model_input({
            **frame,
            "droid_shoulder_light_randomization": frame[external],
            "wrist_camera_zed_mini": frame[wrist],
        })


class RemoteDroidPolicyConfig(PiPolicyConfig):
    camera_names: list[str] | None = None

    def model_post_init(self, context):
        super().model_post_init(context)
        self.policy_cls = RemoteDroidPolicy
        self.policy_factory = make_lenient(RemoteDroidPolicy)


class RemoteDroidEvalConfig(PiPolicyEvalConfig):
    policy_config: RemoteDroidPolicyConfig = RemoteDroidPolicyConfig(
        remote_config={"host": "127.0.0.1", "port": int(os.environ.get("MOLMO_POLICY_PORT", "8080"))}
    )
