"""Authorized E04 results -> E05 full stereo cache -> GPU smoke -> fresh training."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
MAIN=Path('/data/fcy/code/openwam-exp')
QUEUE=MAIN/'reports/E05_training'
PREVIOUS=MAIN/'reports/E04_wan22/automatic_evaluation'
PY='/data/fcy/basics/miniconda3/envs/openwam/bin/python'
sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf
from openwam.experiments import resolve_experiment,preflight,source_fingerprint
from benchmarks.vla_harness.run_resumable import atomic_json,resources_idle
from benchmarks.vla_harness.checkpoint_contract import validate_vggt_omega_checkpoint


def status(state,**details):
    value=dict(state=state,updated_at=time.time(),workspace=str(ROOT),**details)
    atomic_json(QUEUE/'status.json',value)
    path=MAIN/'reports/E05_patch_only/status.json'
    saved=json.loads(path.read_text());saved.update(state=state,updated_at=value['updated_at'],pipeline_status=str(QUEUE/'status.json'))
    if state=='formal_training':saved.update(formal_training_started=True,training_pid=details.get('pid'))
    atomic_json(path,saved)
    print(json.dumps(value),flush=True)


def previous_complete():
    if not (PREVIOUS/'DONE').exists() or not (PREVIOUS/'RESULT_DETAILS_DONE').exists():return False
    expected={'libero':2000,'libero-plus':10030}
    accepted={}
    for benchmark,trials in expected.items():
        data=json.loads((PREVIOUS/'evaluation'/benchmark/'summary.json').read_text())
        assert data['complete'] and not data['smoke'] and not data.get('invalid_records',0)
        actual=sum(r['valid_trials'] for r in data['suites']) if benchmark=='libero' else data['overall']['trials']
        assert actual==trials
        accepted[benchmark]=dict(trials=trials,success_rate=data['total_average_success_rate'])
    result=(MAIN/'result.md').read_text()
    assert '<!-- E04_AUTO_RESULT_END -->' in result and '76.02%' in result
    atomic_json(QUEUE/'e04_results_acceptance.json',accepted)
    return True


def wait_idle():
    while not resources_idle('0,1,2,3',range(8990,8998)):
        if (QUEUE/'STOP').exists():raise RuntimeError('Pipeline STOP requested')
        status('waiting_for_resources');time.sleep(15)


def run(stage,command):
    wait_idle()
    if (QUEUE/'STOP').exists():raise RuntimeError('Pipeline STOP requested')
    if stage=='formal_training' and (QUEUE/'formal_training_launch.json').exists():
        raise RuntimeError('Formal training already launched; refusing duplicate or automatic resume')
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
                raise RuntimeError(f'Pipeline STOP requested during {stage}')
            time.sleep(5)
        if child.returncode:raise RuntimeError(f'{stage} exited {child.returncode}; later stages blocked; inspect {logfile}')
    atomic_json(QUEUE/(stage+'_complete.json'),dict(finished_at=time.time(),exit_code=0))
    return logfile


def checks():
    cfg,profile,_=resolve_experiment('E05')
    assert profile['geometry']==dict(clean_latents=2,future_latents=8,channels=2048,tokens=980,
        temporal_design='history_current_plus_8_future_joint_stereo_per_timestamp')
    assert cfg.training.batch_size==64 and cfg.training.num_epochs==10 and cfg.project.seed==42
    assert cfg.training.resume_ckpt_path is None and cfg.training.finetune_ckpt_path is None
    assert cfg.dataloader.history_num_tubelets==1 and cfg.dataloader.video_stride==4
    assert cfg.model.video_backbone.dit_config.num_layers==30 and cfg.model.video_backbone.dit_config.dim==1536
    assert cfg.model.action_backbone.dim==1024
    cpu=json.loads((ROOT/'reports/E05_full_policy_cpu.json').read_text())
    assert cpu['passed'] and cpu['clean_latents']==2 and cpu['tokens']==980 and cpu['reload_checks']['strict_reload']
    cache_dir=Path(cfg.model.video_backbone.encoder.feature_cache_dir)
    estimate=273465*2048*2*196*2
    assert shutil.disk_usage(cache_dir).free>estimate*1.1
    assert not (Path(cfg.training.output_path).exists() and any(Path(cfg.training.output_path).iterdir()))
    plan=dict(encoder='frozen official VGGT-Omega 512 checkpoint; 224 per view',
        temporal_offsets=[-4,0,4,8,12,16,20,24,28,32],clean=2,future=8,tokens=980,frames=273465,
        cache_dir=str(cache_dir),estimated_cache_gib=estimate/2**30,global_batch=256,steps=10690,
        from_scratch=True,workspace=str(ROOT),source_sha256=source_fingerprint())
    atomic_json(QUEUE/'plan.json',plan)
    return cfg,profile,plan


def main(check_only=False):
    QUEUE.mkdir(parents=True,exist_ok=True)
    with (QUEUE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (QUEUE/'formal_training_launch.json').exists():raise RuntimeError('E05 formal launch already exists')
        cfg,profile,plan=checks()
        if check_only:
            print(json.dumps(dict(preflight_passed=True,e04_results_recorded=previous_complete(),**plan)),flush=True);return
        while not previous_complete():
            if (QUEUE/'STOP').exists():return
            status('waiting_for_e04_results');time.sleep(15)
        cache_dir=Path(plan['cache_dir'])
        prior_probe=QUEUE/'gpu_cache_probe_acceptance.json'
        probe=(json.loads(prior_probe.read_text())['ranks'] if prior_probe.exists()
               else [json.loads((cache_dir/f'build_rank{rank}.json').read_text()) for rank in range(4)])
        for r in probe:
            assert r['complete'] and r['verified'] and all(v['bitwise_equal'] for v in r['verified'])
            assert r['window_verified'] and all(v['all_cache_hits'] and v['bitwise_equal'] for v in r['window_verified'])
        atomic_json(QUEUE/'gpu_cache_probe_acceptance.json',dict(passed=True,ranks=probe))
        run('full_cache_build',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            'scripts/precompute_vggt_features.py','--loader-workers','2','--batch-size','8','--write-inflight','8'])
        reports=[json.loads((cache_dir/f'build_rank{rank}.json').read_text()) for rank in range(4)]
        assert sum(r['processed'] for r in reports)==273465
        assert len({r['identity_hash'] for r in reports})==1
        for r in reports:
            assert r['full_cache'] and r['complete'] and r['processed']==r['expected']
            assert len(r['window_verified'])==2 and all(v['bitwise_equal'] and v['all_cache_hits'] for v in r['window_verified'])
        accepted=dict(passed=True,frames=273465,identity_hash=reports[0]['identity_hash'],cache_namespace=reports[0]['cache_namespace'],ranks=reports)
        atomic_json(cache_dir/'COMPLETE.json',accepted);atomic_json(QUEUE/'cache_acceptance.json',accepted)
        assert source_fingerprint()==plan['source_sha256'],'E05 source changed during cache generation'
        preflight(cfg,profile)
        log=run('training_smoke',[PY,'scripts/experiment.py','train','E05','--gpus','0,1,2,3','--smoke-steps','4'])
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
        validate_vggt_omega_checkpoint(checkpoint.parent,checkpoint.name,config)
        atomic_json(QUEUE/'smoke_acceptance.json',dict(passed=True,ranks=metrics,checkpoint=str(checkpoint)))
        assert source_fingerprint()==plan['source_sha256'],'E05 source changed before formal launch'
        formal=Path(cfg.training.output_path)
        assert not (formal.exists() and any(formal.iterdir())),'Refusing overwrite/resume of formal E05 output'
        atomic_json(QUEUE/'formal_acceptance.json',dict(passed=True,from_scratch=True,steps=10690,tokens=980,global_batch=256))
        run('formal_training',[PY,'scripts/experiment.py','train','E05','--gpus','0,1,2,3'])
        status('training_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true')
    try:main(parser.parse_args().check_only)
    except Exception as error:
        QUEUE.mkdir(parents=True,exist_ok=True);status('failed',error=repr(error));raise
