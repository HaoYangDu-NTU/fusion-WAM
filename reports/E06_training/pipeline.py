"""Authorized E06: frozen native feature cache -> full-batch GPU smoke -> fresh training."""
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
MAIN=Path('/data/fcy/code/openwam-exp')
QUEUE=MAIN/'reports/E06_training'
PY='/data/fcy/basics/miniconda3/envs/openwam/bin/python'
sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf
from openwam.experiments import resolve_experiment,preflight,source_fingerprint
from benchmarks.vla_harness.run_resumable import atomic_json,resources_idle
from benchmarks.vla_harness.checkpoint_contract import validate_flux2_framewise_checkpoint
from benchmarks.vla_harness.prepare_libero_plus import prepare_checkpoint


def status(state,**details):
    atomic_json(QUEUE/'status.json',dict(state=state,updated_at=time.time(),workspace=str(ROOT),**details))


def wait_idle():
    while not resources_idle('0,1,2,3',range(8990,8998)):
        if (QUEUE/'STOP').exists():raise RuntimeError('E06 STOP requested')
        status('waiting_for_resources');time.sleep(15)


def run(stage,command):
    wait_idle()
    if (QUEUE/'STOP').exists():raise RuntimeError('E06 STOP requested')
    if stage=='formal_training' and (QUEUE/'formal_training_launch.json').exists():
        raise RuntimeError('Refusing duplicate formal E06 launch')
    logfile=QUEUE/(stage+'.log')
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
         'PYTHONUNBUFFERED':'1','OPENWAM_DISABLE_FUSED_ROPE':'1','OPENWAM_DISABLE_FUSED_ATTENTION':'1',
         'OPENWAM_DISABLE_COMPILED_RMS_NORM':'1'}
    with logfile.open('w') as out:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(QUEUE/(stage+'_launch.json'),dict(command=command,pid=child.pid,started_at=time.time()))
        status(stage,pid=child.pid,log=str(logfile))
        while child.poll() is None:
            if (QUEUE/'STOP').exists():
                os.killpg(child.pid,signal.SIGTERM);child.wait()
                raise RuntimeError(f'E06 STOP during {stage}')
            time.sleep(5)
        if child.returncode:raise RuntimeError(f'{stage} exited {child.returncode}; inspect {logfile}')
    atomic_json(QUEUE/(stage+'_complete.json'),dict(finished_at=time.time(),exit_code=0))
    return logfile


def main():
    QUEUE.mkdir(parents=True,exist_ok=True)
    with (QUEUE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (QUEUE/'formal_training_launch.json').exists():raise RuntimeError('Formal launch already exists')
        cfg,profile,_=resolve_experiment('E06')
        pre=json.loads((ROOT/'reports/E06_training/preflight.json').read_text());assert pre['passed']
        probe_dir=Path('/data/fcy/code/openwam-exp/cache/flux2_libero_probe_batch16')
        probe=[json.loads((probe_dir/f'build_rank{rank}.json').read_text()) for rank in range(4)]
        for r in probe:
            assert r['complete'] and r['verified'] and all(v['bitwise_equal'] for v in r['verified'])
            assert r['window_verified'] and all(v['bitwise_equal'] and v['all_cache_hits'] for v in r['window_verified'])
        atomic_json(QUEUE/'cache_probe_acceptance.json',dict(passed=True,ranks=probe))
        cache_dir=Path(cfg.model.video_backbone.encoder.feature_cache_dir)
        assert cfg.training.batch_size==64 and cfg.training.num_epochs==10 and cfg.project.seed==42
        assert cfg.training.resume_ckpt_path is None and cfg.training.finetune_ckpt_path is None
        assert not (Path(cfg.training.output_path).exists() and any(Path(cfg.training.output_path).iterdir()))
        plan=dict(encoder='official FLUX.2-klein-4B VAE component, BF16, encoder-only',
            temporal_offsets=[-4,0,4,8,12,16,20,24,28,32],clean=2,future=8,tokens=980,frames=273465,
            cache_dir=str(cache_dir),estimated_cache_gib=273465*2*128*14*14*2/2**30,
            global_batch=256,steps=10690,from_scratch=True,source_sha256=source_fingerprint())
        atomic_json(QUEUE/'plan.json',plan)
        run('full_cache_build',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            'scripts/precompute_flux2_features.py','--loader-workers','2','--batch-size','8','--write-inflight','8'])
        reports=[json.loads((cache_dir/f'build_rank{rank}.json').read_text()) for rank in range(4)]
        assert sum(r['processed'] for r in reports)==273465
        assert len({r['identity_hash'] for r in reports})==1
        for r in reports:
            assert r['full_cache'] and r['complete'] and r['processed']==r['expected']
            assert len(r['window_verified'])==2 and all(v['bitwise_equal'] and v['all_cache_hits'] for v in r['window_verified'])
        accepted=dict(passed=True,frames=273465,identity_hash=reports[0]['identity_hash'],cache_namespace=reports[0]['cache_namespace'],ranks=reports)
        atomic_json(cache_dir/'COMPLETE.json',accepted);atomic_json(QUEUE/'cache_acceptance.json',accepted)
        cpu=json.loads((ROOT/'reports/E06_full_policy_cpu.json').read_text())
        assert cpu['passed'] and cpu['tokens']==980 and cpu['reload_checks']['strict_reload']
        assert cpu['frozen_targets_unchanged'] and cpu['projection_updated'] and cpu['encoder_trainable']==0
        assert source_fingerprint()==plan['source_sha256'],'E06 source changed during cache generation'
        preflight(cfg,profile)
        log=run('training_smoke',[PY,'scripts/experiment.py','train','E06','--gpus','0,1,2,3','--smoke-steps','4'])
        text=log.read_text(errors='replace')
        launch_dir=Path(re.search(r'Launch report: ([^\r\n]+)',text)[1])
        metrics=[]
        for rank in range(4):
            rows=[json.loads(line) for line in (launch_dir/'timing'/f'rank{rank}.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows]==[1,2,3,4]
            assert all(math.isfinite(v) for r in rows for v in r['metrics'].values())
            assert any(r['metrics'].get('grad_norm',0)>0 for r in rows)
            metrics.append(rows[-1])
        checkpoint=Path(re.search(r'\[checkpoint\] Saved final: ([^\r\n]+)',text)[1].strip())
        config=OmegaConf.to_container(OmegaConf.load(checkpoint.parent/'config.yaml'),resolve=True)
        validate_flux2_framewise_checkpoint(checkpoint.parent,checkpoint.name,config)
        deploy=prepare_checkpoint(checkpoint.parent,checkpoint.name,ROOT/'reports/E06_training/plus_smoke_deployment',
            MAIN/'cache/text/rynnbrain1p1-9b/libero-plus/avg_all_layers/corrected_text',corrected=True)
        atomic_json(QUEUE/'smoke_acceptance.json',dict(passed=True,ranks=metrics,checkpoint=str(checkpoint),plus_deployment=str(deploy)))
        assert source_fingerprint()==plan['source_sha256'],'E06 source changed before formal launch'
        formal=Path(cfg.training.output_path)
        assert not (formal.exists() and any(formal.iterdir())),'Refusing overwrite/resume of formal output'
        atomic_json(QUEUE/'formal_acceptance.json',dict(passed=True,from_scratch=True,steps=10690,tokens=980,global_batch=256))
        run('formal_training',[PY,'scripts/experiment.py','train','E06','--gpus','0,1,2,3'])
        status('training_complete')


if __name__=='__main__':
    try:main()
    except Exception as error:
        QUEUE.mkdir(parents=True,exist_ok=True);status('failed',error=repr(error));raise
