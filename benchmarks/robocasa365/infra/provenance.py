"""Version and source provenance, independent from model code."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3] / ".benchmark_runtime" / "robocasa365"
PINS={'robocasa':'a07e365c958c4216cd6bbd5f30b47f09a65c6f00',
      'robosuite':'5ce6643f3092639d08f7b0f90ed1c6a84f50552c'}
VERSIONS={'robocasa':'1.0.1','robosuite':'1.5.2','mujoco':'3.3.1','numpy':'2.2.5','lerobot':'0.3.3'}

def provenance():
    import robocasa,robosuite
    result={'python':sys.version,'executable':sys.executable,'packages':{},'repositories':{},
            'rendering':{'MUJOCO_GL':os.environ.get('MUJOCO_GL'),
                         'PYOPENGL_PLATFORM':os.environ.get('PYOPENGL_PLATFORM')}}
    for name in VERSIONS:result['packages'][name]=md.version(name)
    for name,module in [('robocasa',robocasa),('robosuite',robosuite)]:
        path=Path(module.__file__).resolve().parent.parent
        commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(path),'diff','--name-only','HEAD'],text=True).strip()
        result['repositories'][name]={'path':str(path),'commit':commit,'tracked_modifications':dirty.splitlines()}
    return result


def verify_environment():
    data=provenance()
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Official environment requires Python 3.11')
    for name,version in VERSIONS.items():
        if data['packages'][name]!=version:raise RuntimeError(f'{name}: expected {version}, found {data["packages"][name]}')
    for name,commit in PINS.items():
        repo=data['repositories'][name]
        if repo['commit']!=commit or repo['tracked_modifications']:
            raise RuntimeError(f'Unpinned or modified simulator: {name}: {repo}')
    from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY,TASK_SET_REGISTRY
    soup=DATASET_SOUP_REGISTRY['pretrain_human300']
    training={e['task'] for e in soup}
    groups={sp:list(TASK_SET_REGISTRY[sp]) for sp in ('atomic_seen','composite_seen','composite_unseen')}
    if len(soup)!=300 or len(training)!=300 or any(e['split']!='pretrain' or e['source']!='human' for e in soup):
        raise RuntimeError('Unexpected official Human300 registry')
    if [len(x) for x in groups.values()]!=[18,16,16] or len(set(sum(groups.values(),[])))!=50:
        raise RuntimeError('Unexpected official evaluation registry')
    if not set(groups['atomic_seen']+groups['composite_seen'])<=training or set(groups['composite_unseen'])&training:
        raise RuntimeError('Seen/unseen training overlap disagrees with official protocol')
    data['protocol']={'training_tasks':300,'evaluation_tasks':groups,'kitchen_split':'pretrain',
                      'trials_per_task':50,'overall':'mean of 50 task success rates'}
    return data
