"""Bounded long-rollout throughput profiles, always isolated and marked smoke."""
import ast
import json
import os
from pathlib import Path
import re
import random
from collections import Counter, defaultdict
import signal
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/plus_throughput_20261103'
CKPT=ROOT/'reports/E07_E08_plus_20261103/E07/deployment'
PATTERN=re.compile(r'Inference window=([\d.]+)s batches=(\{.*?\}) requests=(\d+) preprocessing=([\d.]+)s generation=([\d.]+)s')


def windows(folder):
    return [[dict(seconds=float(m[1]),batches=ast.literal_eval(m[2]),requests=int(m[3]),generation=float(m[5]))
             for m in PATTERN.finditer(p.read_text())] for p in sorted(folder.glob('model_*.log'))]


def sample(folder):
    row=dict(time=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),cpu=Path('/sys/fs/cgroup/cpu.stat').read_text())
    p=folder/'run.json'
    if p.exists():
        run=json.loads(p.read_text())
        with sqlite3.connect('file:'+str(Path(run['recording_db']['local_path']).resolve())+'?mode=ro',uri=True,timeout=1) as db:
            row['completed']=db.execute('SELECT count(*) FROM episode_results').fetchone()[0]
    return row


def main():
    while json.loads((REPORT/'status.json').read_text()).get('state')!='complete':
        if (REPORT/'controller.log').exists() and 'dynamic256 failed' in (REPORT/'controller.log').read_text():
            raise RuntimeError('Short smoke failed; inspect before steady profiling')
        time.sleep(5)
    sys.path.insert(0,str(ROOT))
    from benchmarks.vla_harness.libero_plus_protocol import task_catalog
    catalog=task_catalog(ROOT.parent/'benchmarks/LIBERO-plus')
    limit=min(Counter(t['suite'] for t in catalog).values())
    eligible=[t for t in catalog if t['task_id']<limit]
    groups=defaultdict(list)
    for task in eligible:groups[task['category']].append(task)
    rng=random.Random(42)
    for group in groups.values():rng.shuffle(group)
    names=sorted(groups)
    selected=[groups[names[i%7]][i//7] for i in range(512)]
    keys={(t['suite'],t['task_id']) for t in selected}
    exclude=REPORT/'steady_exclude.json'
    exclude.write_text(json.dumps([(t['suite'],t['task_id']) for t in eligible if (t['suite'],t['task_id']) not in keys]))
    (REPORT/'steady_selection.json').write_text(json.dumps(dict(tasks=selected,categories=dict(Counter(t['category'] for t in selected))),indent=2))
    for index,(name,workers,batch) in enumerate([('steady128',128,32),('steady192',192,48)]):
        folder=REPORT/name
        if (REPORT/(name+'.json')).exists():continue
        command=[sys.executable,'-m','benchmarks.vla_harness.run_eval','--benchmark','libero-plus',
                 '--ckpt-dir',str(CKPT),'--ckpt-name','checkpoint_step_10690.safetensors','--gpus','0,1,2,3',
                 '--workers',str(workers),'--max-batch-size',str(batch),'--render-gpus','0,1,2,3',
                 '--base-port',str(9090+index*10),'--correct-instruction-suffix','true',
                 '--smoke','--smoke-max-steps','700','--dynamic-tasks',
                 '--task-ids',','.join(map(str,range(limit))), '--exclude-tasks-file',str(exclude),'--output-dir',str(folder)]
        (REPORT/'steady_status.json').write_text(json.dumps(dict(state='running',case=name,command=command)))
        started=time.monotonic(); steady=None; stopped=False
        with (REPORT/(name+'.log')).open('w') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            while p.poll() is None:
                data=windows(folder)
                if len(data)==4 and all(data) and steady is None:steady=time.monotonic()
                if (steady and time.monotonic()-steady>=180) or time.monotonic()-started>600:
                    p.send_signal(signal.SIGINT); stopped=True;break
                try:
                    if folder.exists():
                        with (REPORT/(name+'_telemetry.jsonl')).open('a') as f:f.write(json.dumps(sample(folder))+'\n')
                except (OSError,ValueError,sqlite3.Error):pass
                time.sleep(5)
            code=p.wait(timeout=90)
        valid=(steady is not None and stopped and code in (0,1,130,-signal.SIGINT)
               and not (folder/'episode_failure.json').exists() and not (folder/'process_failure.json').exists())
        rows=[]
        for stream in windows(folder):
            stream=stream[1:]  # Exclude the first cold window on each GPU.
            requests=sum(w['requests'] for w in stream)
            count=sum(sum(w['batches'].values()) for w in stream)
            rows.append(dict(requests=requests,mean_batch=requests/count if count else None,
                             requests_per_second=requests/sum(w['seconds'] for w in stream) if stream else None))
        result=dict(valid=valid,smoke=True,not_a_formal_score=True,exit_code=code,interrupted_by_profile=stopped,
                    workers=workers,max_batch_size=batch,wall_seconds=time.monotonic()-started,
                    measured_seconds=time.monotonic()-steady if steady else 0,models=rows)
        (REPORT/(name+'.json')).write_text(json.dumps(result,indent=2))
        if not valid:raise RuntimeError(f'{name} invalid; inspect logs')
    (REPORT/'steady_status.json').write_text(json.dumps(dict(state='complete')))


if __name__=='__main__':main()
