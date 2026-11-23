"""Official Gym/registry rollout, pretrain kitchens, no policy success feedback."""
from collections import defaultdict, deque
import csv
import importlib
import json
from pathlib import Path
import numpy as np
from .schema import action_to_env, observation_from_env, CAMERAS

SPLITS=('atomic_seen','composite_seen','composite_unseen')


def summarize(rows):
    tasks={}
    for row in rows:
        tasks.setdefault((row['split'],row['task']),[]).append(float(row['success']))
    rates={key:float(np.mean(values)) for key,values in tasks.items()}
    splits={sp:float(np.mean([v for (s,_),v in rates.items() if s==sp]))
            for sp in ('atomic_seen','composite_seen','composite_unseen') if any(s==sp for s,_ in rates)}
    return {'split_success_rates':splits,
            'task_success_rates':{task:v for (_,task),v in rates.items()},
            'overall_split_mean':float(np.mean(list(splits.values()))) if len(splits)==3 else None,
            'overall_task_mean':float(np.mean(list(rates.values()))) if rates else None,
            'episodes':len(rows),'tasks':len(tasks),'successes':sum(int(x['success']) for x in rows)}


def _history(frames,count):
    values=list(frames);valid=np.asarray([False]*(count-len(values))+[True]*len(values))
    values=[values[0]]*(count-len(values))+values
    return {'rgb':{c:np.concatenate([v['rgb'][c] for v in values]) for c in CAMERAS},
            'state':np.concatenate([v['state'] for v in values]),
            'timestamp':np.concatenate([v['timestamp'] for v in values]),
            'language':values[-1]['language'],'history_valid':valid}


def rollout(env,policy,*,seed,horizon,history=1,fps=20,capture_rgb=False):
    obs,_=env.reset(seed=seed);policy.reset();frames=deque(maxlen=history)
    frames.append(observation_from_env(obs,0));steps=0;success=False;done=False;truncated=False;calls=0;chunk_lengths=set()
    while steps<horizon:
        chunk=np.asarray(policy.predict(_history(frames,history)))
        if chunk.ndim!=2 or chunk.shape[1]!=12 or chunk.shape[0]<1:
            raise ValueError('Policy output must be [K,12], K>=1')
        calls+=1;chunk_lengths.add(int(chunk.shape[0]))
        # No clipping, smoothing, action ensembling, retries or privileged observations.
        for action in chunk:
            obs,_,done,truncated,info=env.step(action_to_env(action))
            steps+=1;success=bool(info.get('success',False))
            frames.append(observation_from_env(obs,steps/fps))
            if success or done or truncated or steps>=horizon:break
        if success or done or truncated:break
    result={'success':success,'steps':steps,'terminated':bool(done),'truncated':bool(truncated),
            'policy_calls':calls,'action_chunk_lengths':sorted(chunk_lengths)}
    if capture_rgb:
        result['_rgb']={c:frames[-1]['rgb'][c][-1] for c in CAMERAS}
    return result


def evaluate(output,*,policy_factory,trials=50,history=1,seed=0,smoke=False,smoke_steps=2,tasks=None,policy_name=None):
    from .provenance import verify_environment, provenance
    verify_environment()
    import gymnasium as gym
    import robocasa.wrappers.gym_wrapper  # official registration
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    if trials<1 or history<1:raise ValueError('trials/history must be positive')
    if not smoke and (trials!=50 or tasks is not None):
        raise ValueError('Standard evaluation requires all 50 tasks and 50 rollouts/task')
    if smoke and smoke_steps<1:raise ValueError('smoke_steps must be positive')
    official={task:sp for sp in SPLITS for task in TASK_SET_REGISTRY[sp]}
    if tasks and set(tasks)-set(official):raise ValueError('Non-official task selection')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'episodes.jsonl').exists():raise FileExistsError('Use a fresh output directory to avoid mixed runs')
    rows=[];meta={'mode':'smoke_non_benchmark' if smoke else 'standard', 'kitchen_split':'pretrain',
                  'trials':trials,'history':history,'seed':seed,
                  'schema_version':'robocasa365.raw.v1',
                  'policy':policy_name or getattr(policy_factory,'__qualname__',type(policy_factory).__name__),
                  'provenance':provenance()}
    (output/'run.json').write_text(json.dumps(meta,indent=2))
    for sp in SPLITS:
        names=[t for t in TASK_SET_REGISTRY[sp] if not tasks or t in tasks]
        if smoke and tasks is None:names=names[:1]
        for task in names:
            horizon=int(get_task_horizon(task));limit=min(horizon,smoke_steps) if smoke else horizon
            env=gym.make('robocasa/'+task,split='pretrain',enable_render=True,seed=seed)
            policy=None
            try:
                policy=policy_factory()
                for trial in range(trials):
                    row={'split':sp,'task':task,'trial':trial,'seed':seed+trial,'official_horizon':horizon,'limit':limit}
                    row.update(rollout(env,policy,seed=seed+trial,horizon=limit,history=history,capture_rgb=smoke))
                    rgb=row.pop('_rgb',None)
                    # Render actually generated pixels, preserve one image per task for smoke inspection.
                    if smoke and trial==0:
                        from PIL import Image
                        row['rgb_statistics']=rendered_rgb_stats(rgb)
                        for camera,image in rgb.items():
                            Image.fromarray(image).save(output/(task+'_'+camera+'.png'))
                        Image.fromarray(env.render()).save(output/(task+'.png'))
                    rows.append(row)
                    with (output/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True)
            finally:
                env.close()
                if policy is not None and hasattr(policy,'close'):policy.close()
    summary={**meta,**summarize(rows)}
    summary['leaderboard_score']=None if smoke else summary['overall_task_mean']
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    with (output/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return summary


def rendered_rgb_stats(rgb):
    """Smoke-only check; never filter/retry/alter observations in standard eval."""
    stats={camera:{'min':int(image.min()),'max':int(image.max()),
                   'mean':float(image.mean()),'std':float(image.std())}
           for camera,image in rgb.items()}
    if not any(v['std']>1 and v['mean']>1 for v in stats.values()):
        raise RuntimeError(f'All camera outputs are blank or nearly blank: {stats}')
    return stats
