"""FM-VJEPA protocol helpers; no model/simulator imports on the client path."""
from pathlib import Path
import hashlib
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FM_ROOT = ROOT / 'diagnose_model/future_prediction_analysis/fm_vjepa_policy'
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def add_policy_arguments(parser):
    parser.add_argument('--policy-backend', choices=('openwam', 'fm-vjepa', 'vpp2'), default='openwam')
    parser.add_argument('--fm-python', type=Path)
    parser.add_argument('--fm-encoder', type=Path)
    parser.add_argument('--fm-text-cache', type=Path)


def fm_arguments(args):
    result = ['--policy-backend', 'fm-vjepa']
    for key in ('fm_python', 'fm_encoder', 'fm_text_cache'):
        value = getattr(args, key, None)
        if value is None or not value.exists():
            raise FileNotFoundError(f'FM evaluation requires an existing --{key.replace("_", "-")}')
        # Resolving a venv interpreter symlink bypasses its pyvenv.cfg.
        path = value.absolute() if key == 'fm_python' else value.resolve()
        result.extend(['--' + key.replace('_', '-'), str(path)])
    return result


def native_state(obs):
    # Match the native VJEPA LIBERO client; preserve BOTH gripper joint positions.
    quat = np.asarray(obs['robot0_eef_quat'], dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1., 1.)
    den = np.sqrt(1. - quat[3] * quat[3])
    aa = np.zeros(3) if math.isclose(den, 0.) else quat[:3] * 2. * math.acos(quat[3]) / den
    state = np.concatenate((obs['robot0_eef_pos'], aa, obs['robot0_gripper_qpos'])).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('FM-VJEPA requires finite native 8D proprioception')
    return state


def text_path(cache_dir, language):
    digest = hashlib.sha256(PROMPT.format(task=language).encode('utf-8')).hexdigest()
    return Path(cache_dir) / f'{digest}.t5_len128.pt'


def check_text_cache(cache_dir, tasks, languages):
    selected = {(t['suite'], t['task_id']) for t in tasks}
    described = {(t['suite'], t['task_id']) for t in languages}
    if not selected <= described:
        raise ValueError('FM language catalog is missing selected tasks')
    prompts = {t['language'] for t in languages if (t['suite'], t['task_id']) in selected}
    missing = [p for p in prompts if not text_path(cache_dir, p).is_file()]
    if missing:
        raise FileNotFoundError(f'FM T5-128 cache missing {len(missing)}/{len(prompts)} prompts; example: {missing[0]!r}')
    return {'mode': 'fm-t5-v1.1-xxl-128', 'directory': str(cache_dir), 'unique_prompts': len(prompts), 'missing': 0}
