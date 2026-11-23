"""Contract tests: episode isolation, native mapping, score weighting and dry-run safety."""
import numpy as np
import pytest
import torch
from .schema import action_to_env, observation_from_env, STATE_FIELDS, ACTION_FIELDS
from .dataset import window_indices
from .evaluation import summarize
from .training import DummyPolicy, dry_run, save_checkpoint, load_checkpoint


def test_history_and_chunk_never_cross_episode():
    idx, valid = window_indices(0, 3, 4, past=True)
    assert idx.tolist() == [0, 0, 0, 0]
    assert valid.tolist() == [False, False, False, True]
    idx, valid = window_indices(2, 3, 4, past=False)
    assert idx.tolist() == [2, 2, 2, 2]
    assert valid.tolist() == [True, False, False, False]
    with pytest.raises(ValueError):
        window_indices(0, 0, 2, past=False)


def test_native_mapping_preserves_values_and_rejects_invalid_actions():
    values = np.arange(12, dtype=np.float32) / 12
    mapped = action_to_env(values)
    assert np.array_equal(mapped['action.base_motion'], values[:4])
    assert np.array_equal(mapped['action.end_effector_position'], values[5:8])
    assert np.array_equal(mapped['action.end_effector_rotation'], values[8:11])
    assert np.array_equal(mapped['action.gripper_close'], values[11:])
    for bad in [np.zeros(15), np.full(12, np.nan), np.full(12, 1.1)]:
        with pytest.raises(ValueError):
            action_to_env(bad)
    obs = {'state.' + name: np.arange(end-start, dtype=np.float32) for name, start, end in STATE_FIELDS}
    obs.update({'video.'+c: np.full((4,4,3), 87, np.uint8) for c in ['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']})
    obs['annotation.human.task_description'] = 'Raw instruction.'
    unified = observation_from_env(obs, timestamp=0.05)
    assert unified['state'].shape == (1, 16)
    assert unified['language'] == 'Raw instruction.'
    assert unified['rgb']['robot0_eye_in_hand'].dtype == np.uint8


def test_scores_export_split_macro_and_task_weighted_separately():
    rows = [{'split':sp, 'task':f'{sp}{i}', 'success':sp=='atomic_seen'}
            for sp,n in [('atomic_seen',18),('composite_seen',16),('composite_unseen',16)] for i in range(n)]
    result = summarize(rows)
    assert result['overall_split_mean'] == pytest.approx(1/3)
    assert result['overall_task_mean'] == pytest.approx(18/50)
    assert summarize(rows[:1])['overall_split_mean'] is None


def test_dummy_dry_run_and_checkpoint_do_not_train(tmp_path):
    policy = DummyPolicy(chunk=2)
    batch = {'observation':{'state':torch.zeros(2,3,16)}, 'action':torch.ones(2,2,12), 'action_valid':torch.tensor([[True,False],[True,True]])}
    result = dry_run(policy, [batch], max_batches=1)
    assert result['batches'] == 1
    assert result['valid_action_steps'] == 3
    path = tmp_path/'dummy.pt'
    save_checkpoint(path, policy, config={'history':3,'chunk':2}, step=0)
    meta = load_checkpoint(path, policy)
    assert meta['step'] == 0 and meta['schema_version'] == 'robocasa365.raw.v1'
    assert list(policy.parameters()) == []


def test_official_archive_allows_readme_and_rejects_escape(tmp_path):
    import io
    import tarfile
    from .data_admin import extract_archive
    archive=tmp_path/'official.tar'
    with tarfile.open(archive,'w') as tar:
        for name in ['lerobot/meta/info.json','README.md']:
            content=b'{}';member=tarfile.TarInfo(name);member.size=len(content)
            tar.addfile(member,io.BytesIO(content))
    extract_archive(archive,tmp_path/'stage')
    assert (tmp_path/'stage/README.md').read_bytes()==b'{}'
    assert (tmp_path/'stage/lerobot/meta/info.json').exists()
    with tarfile.open(archive,'w') as tar:
        member=tarfile.TarInfo('../escape');member.size=2;tar.addfile(member,io.BytesIO(b'{}'))
    with pytest.raises(ValueError):extract_archive(archive,tmp_path/'bad')

    with tarfile.open(archive,'w') as tar:
        content=b'x';member=tarfile.TarInfo('lerobot/../../escape');member.size=1
        tar.addfile(member,io.BytesIO(content))
    with pytest.raises(ValueError):extract_archive(archive,tmp_path/'malicious')
    assert not (tmp_path/'escape').exists()


def test_closed_loop_resets_policy_and_stops_on_official_success():
    from .evaluation import rollout
    class Env:
        def obs(self):
            d={'state.'+name:np.zeros(end-start) for name,start,end in STATE_FIELDS}
            d.update({'video.'+c:np.full((4,4,3),self.steps,np.uint8)
                      for c in ['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']})
            d['annotation.human.task_description']='Do the task.'
            return d
        def reset(self,seed):self.steps=0;return self.obs(),{'success':False}
        def step(self,action):
            assert set(action)=={'action.'+name for name,_,_ in ACTION_FIELDS}
            self.steps+=1
            return self.obs(),0,False,False,{'success':self.steps==3,'privileged':'must not leak'}
    class Policy:
        def reset(self):self.observations=[]
        def predict(self,obs):
            assert 'privileged' not in obs and 'success' not in obs
            self.observations.append(obs)
            return np.zeros((2,12))
    policy=Policy();env=Env()
    assert rollout(env,policy,seed=7,horizon=10,history=3)['steps']==3
    assert len(policy.observations)==2
    assert policy.observations[0]['history_valid'].tolist()==[False,False,True]
    assert policy.observations[1]['timestamp'].tolist()==pytest.approx([0,.05,.1])
    assert rollout(env,policy,seed=8,horizon=1,history=3)['success'] is False
    assert len(policy.observations)==1


def test_dataset_fails_before_iteration_on_missing_payload_or_truncated_metadata(tmp_path,monkeypatch):
    import json
    from . import data_admin
    from .dataset import Human300Dataset
    root=tmp_path/'lerobot';(root/'meta').mkdir(parents=True)
    entry={'task':'OpenDrawer','path':str(root),'source':'human','split':'pretrain',
           'filter_key':'100_demos','url':'official-source'}
    monkeypatch.setattr(data_admin,'load_manifest',lambda _: {'dataset_soup':'pretrain_human300','datasets':[entry]})
    info={'total_episodes':2,'total_frames':4,'chunks_size':1000,'fps':20,
          'codebase_version':'v2.1','robot_type':'PandaOmron','total_videos':6,
          'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
          'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}
    (root/'meta/info.json').write_text(json.dumps(info))
    modality={kind:{name:{'original_key':'observation.state' if kind=='state' else 'action','start':start,'end':end}
                    for name,start,end in fields} for kind,fields in [('state',STATE_FIELDS),('action',ACTION_FIELDS)]}
    from .schema import CAMERAS
    modality['video']={c:{'original_key':'observation.images.'+c} for c in CAMERAS}
    (root/'meta/modality.json').write_text(json.dumps(modality))
    episodes=[{'episode_index':i,'length':2} for i in range(2)]
    epfile=root/'meta/episodes.jsonl';epfile.write_text('\n'.join(json.dumps(e) for e in episodes))
    (root.parent/'download.complete.json').write_text(json.dumps({'task':'OpenDrawer','url':'official-source','archive_sha256':'a'*64,'archive_bytes':100}))
    paths=[]
    for i in range(2):
        paths.append(root/f'data/chunk-000/episode_{i:06d}.parquet')
        paths.extend(root/f'videos/chunk-000/observation.images.{c}/episode_{i:06d}.mp4' for c in CAMERAS)
    for p in paths[:-1]:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'payload')
    with pytest.raises(FileNotFoundError):Human300Dataset('unused')
    paths[-1].write_bytes(b'payload')
    assert len(Human300Dataset('unused'))==4
    epfile.write_text(json.dumps(episodes[0])+'\n')
    with pytest.raises(ValueError,match='count'):Human300Dataset('unused')


def test_nonfinite_training_loss_cannot_mutate_model():
    from .training import TrainingPipeline
    class BadModel(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
        def compute_loss(self,batch):return {'loss':self.weight.sum()*float('nan')}
    model=BadModel();optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
    with pytest.raises(ValueError):TrainingPipeline(model,optimizer).step({},update=True)
    assert torch.equal(model.weight,torch.ones(1))
    assert not optimizer.state


def test_render_smoke_rejects_blank_camera_outputs():
    from .evaluation import rendered_rgb_stats
    from .schema import CAMERAS
    rgb={c:np.zeros((256,256,3),np.uint8) for c in CAMERAS}
    with pytest.raises(RuntimeError,match='blank'):rendered_rgb_stats(rgb)
    rgb[CAMERAS[0]][::2]=128
    stats=rendered_rgb_stats(rgb)
    assert stats[CAMERAS[0]]['std']>1
