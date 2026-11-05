"""Sample isolated smoke telemetry; no changes to simulator or model execution."""
import json
from pathlib import Path
import sqlite3
import subprocess
import time

root=Path(__file__).resolve().parents[1]/'reports/plus_throughput_20261103'
while True:
    state=json.loads((root/'status.json').read_text())
    if state['state']=='complete': break
    folder=root/state['case']
    row=dict(time=time.time(),case=state['case'])
    try:
        run=json.loads((folder/'run.json').read_text())
        db=sqlite3.connect('file:'+str(Path(run['recording_db']['local_path']).resolve())+'?mode=ro',uri=True,timeout=1)
        row['completed']=db.execute('SELECT count(*) FROM episode_results').fetchone()[0]
        db.close()
        path=folder/'task_queue.sqlite'
        if path.exists():
            db=sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True,timeout=1)
            row['pending']=db.execute('SELECT count(*) FROM pending').fetchone()[0]
            db.close()
        row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
        row['cpu']=Path('/sys/fs/cgroup/cpu.stat').read_text()
        row['memory']=int(Path('/sys/fs/cgroup/memory.current').read_text())
        with (root/'progress_telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    except (OSError,ValueError,sqlite3.Error):pass
    time.sleep(5)
