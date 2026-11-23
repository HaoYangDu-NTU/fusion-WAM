# Native observation/action mapping

Authority: pinned official `robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json`, `robocasa/wrappers/gym_wrapper.py`, `robocasa/utils/lerobot_utils.py`, and robosuite `controllers/config/robots/default_pandaomron.json`. Intervals below are Python half-open slices.

| Raw state16 slice | Unified field / official Gym field | Semantics |
|---|---|---|
| `[0:3]` | `state.base_position` | Robot base world XYZ |
| `[3:7]` | `state.base_rotation` | Robot base world quaternion XYZW |
| `[7:10]` | `state.end_effector_position_relative` | Achieved EEF XYZ relative to robot base |
| `[10:14]` | `state.end_effector_rotation_relative` | Achieved EEF quaternion XYZW relative to base |
| `[14:16]` | `state.gripper_qpos` | Both original gripper joint positions |

Dataset state16 is preserved as the concatenation above. Simulation maps the same Gym fields back into precisely this order. Neither side converts to rotation6d or reduces the gripper state.

| Raw action12 slice | Official Gym action | Official controller dispatch |
|---|---|---|
| `[0:3]` | `action.base_motion[:3]` | `robot0_base`: native normalized base velocity commands |
| `[3:4]` | `action.base_motion[3:4]` | `robot0_torso`: native torso position-controller input |
| `[4:5]` | `action.control_mode` | `robot0_base_mode`: official threshold at 0.5 to -1/+1 |
| `[5:8]` | `action.end_effector_position` | First three `robot0_right` OSC_POSE delta commands, base reference frame |
| `[8:11]` | `action.end_effector_rotation` | Last three `robot0_right` delta axis-angle commands |
| `[11:12]` | `action.gripper_close` | `robot0_right_gripper`: official threshold at 0.5 to -1/+1 |

These are native controller inputs, **not physical absolute position targets**. The official OSC scales position inputs to ±0.05 and orientation inputs to ±0.5 internally; do not apply this scaling a second time. The adapter copies all twelve source values unchanged into Gym fields. Binary thresholding, base/torso mode behavior, and final control-vector ordering belong to the official Gym wrapper/controller.

RGB mapping is direct:

| Original dataset feature | Official Gym key | Unified RGB key |
|---|---|---|
| `observation.images.robot0_agentview_left` | `video.robot0_agentview_left` | `rgb.robot0_agentview_left` |
| `observation.images.robot0_agentview_right` | `video.robot0_agentview_right` | `rgb.robot0_agentview_right` |
| `observation.images.robot0_eye_in_hand` | `video.robot0_eye_in_hand` | `rgb.robot0_eye_in_hand` |

RGB is uint8 HWC at native 256×256 resolution. The Gym wrapper performs the official renderer vertical flip; the adapter adds no extra flip. LeRobot's official decoder returns CHW float RGB; the dataset adapter restores the decoded integer pixels with round(value×255), then changes layout to HWC. There is no crop/resize/color jitter.

Language comes from `annotation.human.task_description`, resolved through the original `meta/tasks.jsonl` index. It is not inferred from directory names and is not rewritten with a prompt template. The raw `annotation.human.task_name` is not an extra input.

Existing OpenWAM's compact state19/action15 interfaces remain separate. This infrastructure does not silently map native axes into their rotation6d representation, normalize using model statistics, or change existing model code. A later model adapter must explicitly declare and test any model representation conversion.

The official live Gym wrapper supplies float32 proprioception, while original dataset parquet typically stores float64. Both are preserved; a future model adapter explicitly chooses its compute dtype. No implicit normalization is performed.
