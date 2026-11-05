"""Isolated real-model/real-environment smoke; never produces formal scores."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/plus_throughput_20261103'
CKPT = ROOT/'reports/E07_E08_plus_20261103/E07/deployment'
CASES = [('static128',128,32,False), ('dynamic128',128,32,True),
         ('dynamic192',192,48,True), ('dynamic256',256,64,True)]


def summarize(folder, duration):
    rows = []
    for path in sorted(folder.glob('model_*.log')):
        windows = []
        for line in path.read_text().splitlines():
            m = re.search(r'Inference window=([\d.]+)s batches=(\{.*?\}) requests=(\d+) preprocessing=([\d.]+)s generation=([\d.]+)s',line)
            if m:
                windows.append(dict(seconds=float(m[1]),batches=ast.literal_eval(m[2]),requests=int(m[3]),generation=float(m[5])))
        count = sum(sum(w['batches'].values()) for w in windows)
        requests = sum(w['requests'] for w in windows)
        rows.append(dict(server=path.name, windows=len(windows), requests=requests,
                         mean_batch=requests/count if count else None,
                         requests_per_second=requests/sum(w['seconds'] for w in windows) if windows else None))
    summary = json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
    # Count actual episode rows in the harness merged output, including smoke truncation.
    return dict(wall_seconds=duration, models=rows, summary=summary)


def main():
    REPORT.mkdir(exist_ok=True)
    results = {}
    for case_index,(name,workers,batch,dynamic) in enumerate(CASES):
        out = REPORT/name
        if (REPORT/(name+'.json')).exists():
            results[name] = json.loads((REPORT/(name+'.json')).read_text())
            continue
        command = [sys.executable,'-m','benchmarks.vla_harness.run_eval',
                   '--benchmark','libero-plus','--ckpt-dir',str(CKPT),
                   '--ckpt-name','checkpoint_step_10690.safetensors','--gpus','0,1,2,3',
                   '--workers',str(workers),'--max-batch-size',str(batch),
                   '--render-gpus','0,1,2,3','--base-port',str(9010+case_index*10),
                   '--correct-instruction-suffix','true','--smoke',
                   '--task-ids',','.join(map(str,range(128))), '--output-dir',str(out)]
        if dynamic: command += ['--dynamic-tasks']
        (REPORT/'status.json').write_text(json.dumps(dict(state='running',case=name,command=command)))
        started=time.monotonic()
        with (REPORT/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            code=process.wait()
        row = dict(case=name,workers=workers,max_batch_size=batch,dynamic=dynamic,
                   exit_code=code,**summarize(out,time.monotonic()-started))
        (REPORT/(name+'.json')).write_text(json.dumps(row,indent=2))
        results[name]=row
        (REPORT/'results.json').write_text(json.dumps(results,indent=2))
        if code:
            raise RuntimeError(f'{name} failed; inspect log before continuing')
    (REPORT/'status.json').write_text(json.dumps(dict(state='complete')))


if __name__=='__main__': main()
