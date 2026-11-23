"""Allowlisted preparation and explicit integrity reports for official Human300."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.request
from .dataset import read_jsonl
from .schema import CAMERAS,validate_modality


def build_manifest(path):
    from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY, TARGET_TASKS
    from robocasa.scripts.download_datasets import BOX_LINKS_DS, _get_direct_download_url
    from robocasa import macros
    import robocasa
    from .provenance import provenance
    base=Path(macros.DATASET_BASE_PATH) if macros.DATASET_BASE_PATH else Path(robocasa.__file__).resolve().parents[1]/'datasets'
    entries=[]
    for raw in DATASET_SOUP_REGISTRY['pretrain_human300']:
        entry=dict(raw)
        assert entry['split']=='pretrain' and entry['source']=='human'
        relative=Path(entry['path']).relative_to(base/'v1.0')
        key=str(relative.parent/(relative.name+'.tar'))
        if not key.startswith('pretrain/'):raise ValueError('Forbidden source')
        entry['url']=_get_direct_download_url(BOX_LINKS_DS[key],ext='tar')
        entries.append(entry)
    assert len(entries)==300
    data={'dataset_soup':'pretrain_human300','registry_commit':provenance()['repositories']['robocasa']['commit'],
          'datasets':entries,'evaluation_tasks':TARGET_TASKS}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2))
    return data


def load_manifest(path):
    data=json.loads(Path(path).read_text())
    if data['dataset_soup']!='pretrain_human300' or len(data['datasets'])!=300:
        raise ValueError('Expected complete Human300 manifest')
    # Manifest must agree with official registry, including paths and source URLs.
    from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
    expected={e['task']:e for e in DATASET_SOUP_REGISTRY['pretrain_human300']}
    from robocasa.scripts.download_datasets import BOX_LINKS_DS,_get_direct_download_url
    for e in data['datasets']:
        if e['task'] not in expected:raise ValueError('Task outside Human300')
        for key in ['path','split','source','filter_key']:
            if e[key]!=expected[e['task']][key]:raise ValueError(f'Manifest mismatch: {key}')
        # Registry path identifies the exact archive, not just the split name.
        path=Path(e['path']);parts=path.parts;start=parts.index('v1.0')
        archive=str(Path(*parts[start+1:-1])/'lerobot.tar')
        if e['url']!=_get_direct_download_url(BOX_LINKS_DS[archive],ext='tar'):
            raise ValueError('Unapproved download URL')
    if len({e['task'] for e in data['datasets']})!=300:raise ValueError('Duplicate tasks')
    return data


def download_one(entry):
    root=Path(entry['path']);receipt=root.parent/'download.complete.json'
    if receipt.exists() and (root/'meta/info.json').exists():return {'task':entry['task'],'status':'already_present'}
    if root.exists():raise RuntimeError(f'Incomplete unreceipted destination: {root}')
    root.parent.mkdir(parents=True,exist_ok=True);archive=root.parent/'lerobot.tar.part'
    last=None
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(entry['url'],archive);last=None;break
        except Exception as error:last=error;time.sleep(1)
    if last:raise last
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    stage=root.parent/'.extracting';stage.mkdir(exist_ok=True)
    extract_archive(archive,stage)
    # Only a completed extraction is made visible to readers.
    os.replace(stage/'lerobot',root)
    if (stage/'README.md').exists():os.replace(stage/'README.md',root.parent/'README.md')
    stage.rmdir()
    result={'task':entry['task'],'url':entry['url'],'archive_bytes':archive.stat().st_size,'archive_sha256':digest}
    receipt.write_text(json.dumps(result,indent=2));archive.unlink()
    return result


def download_all(manifest,workers=8):
    entries=load_manifest(manifest)['datasets']
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(download_one,entries):print(json.dumps(result),flush=True)


def audit_one(entry,*,deep=False):
    import numpy as np
    import pyarrow.parquet as pq
    import av
    root=Path(entry['path']);info=json.loads((root/'meta/info.json').read_text())
    if info['codebase_version']!='v2.1' or info['fps']!=20 or info['robot_type']!='PandaOmron':
        raise ValueError('Unexpected official format')
    validate_modality(json.loads((root/'meta/modality.json').read_text()))
    episodes=read_jsonl(root/'meta/episodes.jsonl')
    assert len(episodes)==info['total_episodes']
    assert len({e['episode_index'] for e in episodes})==len(episodes)
    assert sum(e['length'] for e in episodes)==info['total_frames']
    tasks={t['task_index']:t['task'] for t in read_jsonl(root/'meta/tasks.jsonl')}
    hashes={};videos=0;frames=0
    for ep in episodes:
        idx=ep['episode_index'];length=ep['length']
        fmt={'episode_chunk':idx//info['chunks_size'],'episode_index':idx}
        path=root/info['data_path'].format(**fmt)
        table=pq.read_table(path);assert len(table)==length
        d=table.to_pydict();state=np.asarray(d['observation.state']);action=np.asarray(d['action'])
        assert state.shape==(length,16) and action.shape==(length,12)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        assert np.all(np.asarray(d['episode_index']).reshape(-1)==idx)
        assert np.array_equal(np.asarray(d['frame_index']).reshape(-1),np.arange(length))
        timestamp=np.asarray(d['timestamp']).reshape(-1)
        assert np.allclose(timestamp,np.arange(length)/info['fps'],atol=1e-4)
        for value in np.asarray(d['annotation.human.task_description']).reshape(-1):assert int(value) in tasks
        frames+=length
        for camera in CAMERAS:
            path=root/info['video_path'].format(**fmt,video_key='observation.images.'+camera)
            with av.open(str(path)) as video:
                stream=video.streams.video[0]
                assert stream.width==256 and stream.height==256
                assert stream.frames==length
                if deep:
                    count=sum(1 for _ in video.decode(video=0));assert count==length
                else:
                    first=next(video.decode(video=0));assert first.width==256
            videos+=1
    assert videos==info['total_videos']
    # Hash every original file including extras, without using privileged extras for training.
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as f:hashes[str(path.relative_to(root))]=hashlib.file_digest(f,'sha256').hexdigest()
    return {'task':entry['task'],'status':'complete','episodes_raw':len(episodes),
            'episodes_train':min(100,len(episodes)),'frames_raw':frames,'videos':videos,
            'decode_mode':'all_frames' if deep else 'headers_and_first_frame',
            'file_sha256':hashes,'official_checksums_available':False}


def audit_all(manifest,output,*,deep=False,workers=4):
    data=load_manifest(manifest);results=[]
    def one(entry):
        try:return audit_one(entry,deep=deep)
        except Exception as error:return {'task':entry['task'],'status':'failed','error':repr(error)}
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(one,data['datasets']):
            results.append(result);print(result['task'],result['status'],result.get('error',''),flush=True)
    report={'dataset_soup':'pretrain_human300','complete':all(r['status']=='complete' for r in results),
            'tasks_complete':sum(r['status']=='complete' for r in results),'tasks_expected':300,'datasets':results}
    Path(output).write_text(json.dumps(report,indent=2));return report


def extract_archive(archive,stage):
    stage=Path(stage);stage.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            p=Path(member.name)
            if (p.is_absolute() or '..' in p.parts or not p.parts or
                (p.parts[0]!='lerobot' and member.name!='README.md') or
                member.issym() or member.islnk() or not (member.isfile() or member.isdir())):
                raise ValueError(f'Unsafe archive member: {member.name}')
            target=stage/p
            if not target.resolve().is_relative_to(stage.resolve()):raise ValueError('Extraction path escape')
            tar.extract(member,stage)
