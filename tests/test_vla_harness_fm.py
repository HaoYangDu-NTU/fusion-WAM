import asyncio
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

from benchmarks.vla_harness.fm_policy import native_state, text_path, check_text_cache
from benchmarks.vla_harness.fm_model_server import FMVJEPAModelServer
from benchmarks.vla_harness.libero import make_observation
from vla_eval.model_servers.predict import PredictModelServer


def raw_obs():
    return {'robot0_eef_pos': np.array([.1,.2,.3]), 'robot0_eef_quat': np.array([0.,0.,0.,1.]),
            'robot0_gripper_qpos': np.array([.025,-.018]),
            'head': np.arange(256*256*3,dtype=np.uint8).reshape(256,256,3),
            'wrist': np.full((256,256,3), 123, dtype=np.uint8)}


def test_raw_state_retains_both_fingers_and_native_axis_angle():
    raw=raw_obs(); raw['robot0_eef_quat']=np.array([0.,0.,np.sin(.3),np.cos(.3)])
    state=native_state(raw)
    np.testing.assert_allclose(state, [.1,.2,.3,0.,0.,.6,.025,-.018],atol=1e-7)


def test_benchmark_fm_pixels_state_and_old_default():
    raw=raw_obs();cfg={'head_camera_key':'head','left_wrist_camera_key':'wrist','image_transform':'rotate_180'}
    old=make_observation(raw,'pick',cfg)
    assert old['state'].shape==(10,)
    fm=make_observation(raw,'pick',{**cfg,'policy_backend':'fm-vjepa'})
    assert fm['state'].shape==(8,)
    np.testing.assert_array_equal(fm['images']['head_camera'], raw['head'][::-1,::-1])
    assert fm['images']['left_wrist_camera'].shape==(256,256,3)
    fm['history_images']=fm['images']
    native=FMVJEPAModelServer.native_observation(fm)
    actual=(native['observation.images.image'][0]*255).round().byte().permute(1,2,0).numpy()
    np.testing.assert_array_equal(actual,raw['head'][::-1,::-1])
    np.testing.assert_array_equal(native['observation.state'][0],fm['state'])


def test_history_per_control_step_session_isolation_reset(monkeypatch):
    captured=[]
    async def capture(self, obs, ctx):captured.append((ctx.session_id,obs['history_images']))
    async def noop(*args):pass
    monkeypatch.setattr(PredictModelServer,'on_observation',capture)
    monkeypatch.setattr(PredictModelServer,'on_episode_start',noop)
    monkeypatch.setattr(PredictModelServer,'on_episode_end',noop)
    server=object.__new__(FMVJEPAModelServer);server._episodes={};server._history={}
    a=SimpleNamespace(session_id='a',episode_id='1');b=SimpleNamespace(session_id='b',episode_id='1')
    async def run():
        await server.on_episode_start(None,a);await server.on_episode_start(None,b)
        for t in range(20):
            await server.on_observation({'images':t},a)
            await server.on_observation({'images':100+t},b)
        await server.on_episode_end(None,a)
        a.episode_id='2';await server.on_episode_start(None,a)
        await server.on_observation({'images':99},a)
    asyncio.run(run())
    assert [v for s,v in captured[:-1] if s=='a']==[max(t-4,0) for t in range(20)]
    assert [v for s,v in captured[:-1] if s=='b']==[100+max(t-4,0) for t in range(20)]
    assert captured[-1]==('a',99)


def test_native_actions_pass_through_without_eef_or_double_gripper_conversion(monkeypatch):
    server=object.__new__(FMVJEPAModelServer)
    actions=np.arange(224,dtype=np.float32).reshape(32,7)/224
    outputs=[{'actions':actions}]
    server.policy=SimpleNamespace(infer_batch=lambda obs:outputs)
    monkeypatch.setattr(server,'native_observation',lambda obs:obs)
    assert server._predict([{}]) is outputs
    np.testing.assert_array_equal(outputs[0]['actions'],actions)


def test_full_prompt_text_cache_and_missing_language(tmp_path):
    tasks=[{'suite':'s','task_id':1}];langs=[{'suite':'s','task_id':1,'language':'pick cup'}]
    with pytest.raises(FileNotFoundError):check_text_cache(tmp_path,tasks,langs)
    text_path(tmp_path,'pick cup').touch()
    assert check_text_cache(tmp_path,tasks,langs)['missing']==0
    assert text_path(tmp_path,'pick cup')!=text_path(tmp_path,'pick cup table 1')
    with pytest.raises(ValueError):check_text_cache(tmp_path,tasks,[])


def test_queue_requires_final_marker_and_both_full_summaries(tmp_path):
    import json
    from benchmarks.vla_harness.run_after import evaluation_complete
    assert not evaluation_complete(tmp_path)
    (tmp_path/'DONE').touch()
    for stage in ('libero','libero-plus'):
        (tmp_path/stage).mkdir()
        (tmp_path/stage/'summary.json').write_text(json.dumps({'selected_run_complete':True,'complete':True,'smoke':False}))
    assert evaluation_complete(tmp_path)
    path=tmp_path/'libero-plus/summary.json'
    path.write_text(json.dumps({'selected_run_complete':False,'smoke':False}))
    assert not evaluation_complete(tmp_path)
    path.write_text(json.dumps({'selected_run_complete':True,'smoke':True}))
    assert not evaluation_complete(tmp_path)


def test_fm_plan_keeps_trials_and_sharding_but_uses_native_server(tmp_path):
    from benchmarks.vla_harness.run_eval import make_plan
    args=SimpleNamespace(policy_backend='fm-vjepa',fm_python=Path('/bin/true'),fm_encoder=tmp_path,
        fm_text_cache=tmp_path,gpus='0,1',suites='libero_spatial',task_ids=None,smoke=False,workers=4,
        max_batch_size=2,max_wait_time=.01,render='gpu',render_gpus='3',base_port=14940,
        ckpt_dir=tmp_path,ckpt_name='checkpoint.pt',output_dir=tmp_path,eval_id='test')
    plan=make_plan(args)
    assert plan['num_trials']==50 and len(plan['tasks'])==10
    for server in plan['servers']:
        cmd=server['command']
        assert cmd[:3]==['/bin/true','-m','benchmarks.vla_harness.fm_model_server']
        assert cmd[cmd.index('--args.checkpoint_name')+1]=='checkpoint.pt'
    assigned=[(t['suite'],t['task_id']) for w in plan['workers'] for t in w['tasks']]
    assert len(assigned)==len(set(assigned))==10


def test_legacy_controller_job_identity_still_resumes(tmp_path,monkeypatch):
    import hashlib,json,sys
    from benchmarks.vla_harness import run_resumable as resume
    (tmp_path/'config.yaml').write_text('model: {}\n')
    (tmp_path/'model.safetensors').write_bytes(b'weights')
    weight=(tmp_path/'model.safetensors').stat()
    settings={'ckpt_dir':str(tmp_path),'ckpt_name':'model.safetensors','output_dir':str(tmp_path),
        'gpus':'0,1,2,3','workers':32,'max_batch_size':8,'render_gpus':'3','base_port':8940,
        'retry_delay':30,'config_sha256':hashlib.sha256((tmp_path/'config.yaml').read_bytes()).hexdigest(),
        'weight_size_mtime':[weight.st_size,weight.st_mtime_ns]}
    (tmp_path/'job.json').write_text(json.dumps({'job_id':'legacy','settings':settings}))
    # Legacy jobs predate instruction correction; explicitly retain that
    # protocol instead of inheriting the newer default on resume.
    monkeypatch.setattr(sys,'argv',['resume','--ckpt-dir',str(tmp_path),'--ckpt-name','model.safetensors','--output-dir',str(tmp_path),
        '--correct-instruction-suffix','false'])
    monkeypatch.setattr(resume,'stop_owned_processes',lambda _:None)
    stages=[]
    monkeypatch.setattr(resume,'run_stage',lambda a,j,b:stages.append(b) or True)
    resume.main()
    assert stages==['libero','libero-plus']
    assert json.loads((tmp_path/'job.json').read_text())['settings']==settings


def test_fm_controller_preserves_venv_interpreter_symlink(tmp_path):
    from benchmarks.vla_harness.fm_policy import fm_arguments
    base = tmp_path / 'base-python'
    base.touch()
    interpreter = tmp_path / 'venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(base)
    args = SimpleNamespace(fm_python=interpreter, fm_encoder=tmp_path, fm_text_cache=tmp_path)
    command = fm_arguments(args)
    selected = command[command.index('--fm-python') + 1]
    assert selected == str(interpreter.absolute())
    assert selected != str(interpreter.resolve())
