"""Unaltered official PandaOmron LeRobot modality contract (v2.1).

Quaternions are xyzw, EEF state is relative to robot base; base state is world.
EEF actions are normalized OSC delta position/axis-angle, not absolute poses.
The official Gym converter owns binary gripper/control-mode thresholding.
"""
from typing import Protocol
import numpy as np

SCHEMA_VERSION = 'robocasa365.raw.v1'
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
STATE_FIELDS = (('base_position',0,3), ('base_rotation',3,7),
                ('end_effector_position_relative',7,10),
                ('end_effector_rotation_relative',10,14), ('gripper_qpos',14,16))
ACTION_FIELDS = (('base_motion',0,4), ('control_mode',4,5),
                 ('end_effector_position',5,8), ('end_effector_rotation',8,11),
                 ('gripper_close',11,12))

class Policy(Protocol):
    def reset(self) -> None: ...
    def predict(self, observation: dict) -> np.ndarray:
        """Return [K,12] actions in this schema, K>=1. No privileged inputs."""
        ...


def validate_modality(meta):
    for kind, fields in [('state', STATE_FIELDS), ('action', ACTION_FIELDS)]:
        for name, start, end in fields:
            field = meta[kind][name]
            expected_key = 'observation.state' if kind == 'state' else 'action'
            if (field['original_key'],field['start'],field['end']) != (expected_key,start,end):
                raise ValueError(f'Unsupported {kind}.{name} metadata: {field}')
    for camera in CAMERAS:
        if meta['video'][camera]['original_key'] != 'observation.images.'+camera:
            raise ValueError(f'Unsupported camera metadata: {camera}')


def action_to_env(action):
    a = np.asarray(action)
    if a.shape != (12,) or not np.isfinite(a).all() or np.any(np.abs(a)>1):
        raise ValueError('Expected finite normalized native action [12] in [-1,1]')
    return {'action.'+name:a[start:end].copy() for name,start,end in ACTION_FIELDS}


def observation_from_env(obs, timestamp):
    state = np.concatenate([obs['state.'+name] for name,_,_ in STATE_FIELDS])
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError('Official Gym proprioception must be finite state16')
    return {'rgb':{c:np.asarray(obs['video.'+c],dtype=np.uint8)[None].copy() for c in CAMERAS},
            'state':state[None].copy(), 'language':obs['annotation.human.task_description'],
            'timestamp':np.asarray([timestamp],dtype=np.float64),
            'history_valid':np.ones(1,dtype=bool)}
