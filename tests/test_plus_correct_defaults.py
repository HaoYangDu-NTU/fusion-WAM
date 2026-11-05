import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.vla_harness import run_eval, run_resumable, prepare_libero_plus_svae

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('benchmark,override,expected', [
    ('libero',None,False),('libero-plus',None,True),('libero-plus','false',False),
    ('libero-plus','true',True),('libero','false',False)])
def test_cli_default_is_benchmark_specific(monkeypatch, benchmark, override, expected):
    args=['run_eval','--benchmark',benchmark,'--dry-run']
    if override is not None:args+=['--correct-instruction-suffix',override]
    monkeypatch.setattr(sys,'argv',args)
    def plan(options):
        assert options.correct_instruction_suffix is expected
        return {}
    monkeypatch.setattr(run_eval,'make_plan',plan)
    assert run_eval.main()==0


def test_standard_libero_rejects_true(monkeypatch):
    monkeypatch.setattr(sys,'argv',['run_eval','--benchmark','libero','--correct-instruction-suffix','true','--dry-run'])
    with pytest.raises(SystemExit):run_eval.main()


@pytest.mark.parametrize('override,expected',[(None,True),('false',False)])
def test_resumable_default_and_explicit_legacy_identity(tmp_path,monkeypatch,override,expected):
    (tmp_path/'config.yaml').write_text('model: {}')
    (tmp_path/'model.safetensors').write_bytes(b'test')
    output=tmp_path/'output'
    args=['run_resumable','--ckpt-dir',str(tmp_path),'--ckpt-name','model.safetensors','--output-dir',str(output)]
    if override is not None:args+=['--correct-instruction-suffix',override]
    monkeypatch.setattr(sys,'argv',args)
    monkeypatch.setattr(run_resumable,'stop_owned_processes',lambda *_:None)
    calls=[]
    def stage(options,*_):
        calls.append(options.correct_instruction_suffix)
        return False
    monkeypatch.setattr(run_resumable,'run_stage',stage)
    run_resumable.main()
    assert calls==[expected]
    settings=json.loads((output/'job.json').read_text())['settings']
    assert settings.get('correct_instruction_suffix',False) is expected
    if not expected:
        assert 'correct_instruction_suffix' not in settings
        monkeypatch.setattr(sys,'argv',args[:-2])
        with pytest.raises(ValueError,match='Resume settings'):
            run_resumable.main()


@pytest.mark.parametrize('benchmark,override,expected', [('libero',None,'False'),('libero-plus',None,'True'),('libero-plus','false','False')])
def test_shell_passes_same_value_to_prepare_and_eval(tmp_path,benchmark,override,expected):
    fake=tmp_path/'python';log=tmp_path/'calls'
    fake.write_text('#!'+sys.executable+'\nimport os,sys,json\nwith open(os.environ["CALLS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    args=['bash',str(ROOT/'scripts/eval_libero_vjepa.sh'),str(tmp_path),'model.safetensors','--benchmark',benchmark,'--dry-run']
    if override is not None:args+=['--correct-instruction-suffix',override]
    env={**os.environ,'OPENWAM_PYTHON':str(fake),'CALLS':str(log)}
    env.pop('CORRECT_INSTRUCTION_SUFFIX',None)
    subprocess.run(args,env=env,check=True,timeout=10)
    calls=[json.loads(s) for s in log.read_text().splitlines()]
    assert len(calls)==(2 if benchmark=='libero-plus' else 1)
    for call in calls:assert call[call.index('--correct-instruction-suffix')+1]==expected


@pytest.mark.parametrize("override,expected",[(None,True),("false",False),("true",True)])
def test_prepare_cli_defaults_to_corrected(monkeypatch,tmp_path,override,expected):
    monkeypatch.setattr(sys,'argv',['prepare','--source-dir',str(tmp_path),'--ckpt-name','model','--output-dir',str(tmp_path/'out')])
    if override is not None:sys.argv.extend(['--correct-instruction-suffix',override])
    captured=[]
    monkeypatch.setattr(prepare_libero_plus_svae,'prepare_checkpoint',lambda *a,**kw:captured.append(kw['corrected']))
    prepare_libero_plus_svae.main()
    assert captured==[expected]
