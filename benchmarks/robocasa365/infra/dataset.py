"""Read-only official LeRobot v2.1 episode windows; native values are preserved.

Uses PyArrow for the episode parquet and LeRobot's video decoder. This avoids
loading every task into HuggingFace datasets or fetching data implicitly.
"""
from bisect import bisect_right
from functools import lru_cache
import json
from pathlib import Path
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .schema import CAMERAS, validate_modality


def window_indices(frame, length, count, *, past):
    if length<1 or count<1 or not 0<=frame<length:
        raise ValueError('Invalid episode length, frame or window size')
    raw = frame + (np.arange(1-count,1) if past else np.arange(count))
    return np.clip(raw,0,length-1), (raw>=0)&(raw<length)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class Human300Dataset(Dataset):
    """Manifest-only allowlist. Full loading fails if any selected task is missing.

    The official 100_demos filter uses Random(0).shuffle, with all raw episodes
    retained on disk. No resampling/weighting beyond flat frame indexing.
    """
    def __init__(self, manifest, *, history=1, chunk=1, tasks=None):
        from .data_admin import load_manifest
        self.manifest=load_manifest(manifest)
        if self.manifest['dataset_soup']!='pretrain_human300':
            raise ValueError('Only pretrain_human300 is allowed')
        if history<1 or chunk<1: raise ValueError('history/chunk must be positive')
        self.history,self.chunk=history,chunk
        entries=self.manifest['datasets']
        if tasks is not None:
            unknown=set(tasks)-{e['task'] for e in entries}
            if unknown:raise ValueError(f'Tasks outside Human300: {unknown}')
            entries=[e for e in entries if e['task'] in tasks]
        if not entries:raise ValueError('No datasets selected')
        self.episodes=[];self.ends=[];total=0
        for entry in entries:
            if entry['split']!='pretrain' or entry['source']!='human':
                raise ValueError('Forbidden training source')
            root=Path(entry['path']);info=json.loads((root/'meta/info.json').read_text())
            validate_modality(json.loads((root/'meta/modality.json').read_text()))
            episodes=read_jsonl(root/'meta/episodes.jsonl')
            validate_task_layout(entry,info,episodes)
            # Exact algorithm in official get_subset_demos_filter_key, no global RNG mutation.
            ids=[e['episode_index'] for e in episodes];random.Random(0).shuffle(ids)
            selected=set(ids[:int(entry['filter_key'].split('_')[0])])
            for ep in episodes:
                if ep['episode_index'] not in selected:continue
                length=int(ep['length'])
                if length<1:raise ValueError(f'Empty episode: {root} {ep}')
                self.episodes.append((entry,info,ep));total+=length;self.ends.append(total)

    def __len__(self):return self.ends[-1]

    @staticmethod
    @lru_cache(maxsize=4)
    def _episode_table(path):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pydict()

    def __getitem__(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        e=bisect_right(self.ends,index);entry,info,ep=self.episodes[e]
        frame=index-(self.ends[e-1] if e else 0);length=int(ep['length'])
        hist,hvalid=window_indices(frame,length,self.history,past=True)
        future,avalid=window_indices(frame,length,self.chunk,past=False)
        fmt={'episode_chunk':ep['episode_index']//info['chunks_size'],'episode_index':ep['episode_index']}
        root=Path(entry['path']);data=self._episode_table(str(root/info['data_path'].format(**fmt)))
        if len(data['action'])!=length:raise ValueError('Episode length mismatch')
        states=np.asarray(data['observation.state']);actions=np.asarray(data['action'])
        times=np.asarray(data['timestamp']).reshape(-1)
        if states.shape!=(length,16) or actions.shape!=(length,12):raise ValueError('Non-native dimensions')
        # Annotation task IDs are not necessarily equal to the generic task_index.
        language_id=int(np.asarray(data['annotation.human.task_description'][frame]).item())
        languages={x['task_index']:x['task'] for x in read_jsonl(root/'meta/tasks.jsonl')}
        from lerobot.datasets.video_utils import decode_video_frames
        rgb={}
        for camera in CAMERAS:
            video_key='observation.images.'+camera
            path=root/info['video_path'].format(**fmt,video_key=video_key)
            decoded=decode_video_frames(path, times[hist].tolist(), tolerance_s=1/info['fps']/2+1e-6, backend='pyav')
            # LeRobot returns float CHW RGB [0,1]; round-trip to decoded uint8 HWC.
            rgb[camera]=(decoded.permute(0,2,3,1)*255).round().to(dtype=__import__('torch').uint8).numpy()
        return {'observation':{'rgb':rgb,'state':states[hist].copy(),
                    'language':languages[language_id], 'timestamp':times[hist].copy(), 'history_valid':hvalid},
                'action':actions[future].copy(),'action_valid':avalid,'action_timestamp':times[future].copy(),
                'episode_id':f"{entry['task']}/{ep['episode_index']}",'task':entry['task'],
                'frame_index':frame,'episode_length':length,'is_first':frame==0,'is_last':frame==length-1,
                'next_done':np.asarray(data['next.done'])[future].reshape(-1).copy()}


def make_loader(dataset, *, batch_size=1, workers=0, shuffle=False):
    return DataLoader(dataset,batch_size=batch_size,num_workers=workers,shuffle=shuffle)


def validate_task_layout(entry,info,episodes):
    """Fail before exposing any frames if a declared source task is incomplete."""
    root=Path(entry['path'])
    if info.get('codebase_version')!='v2.1' or info.get('fps')!=20 or info.get('robot_type')!='PandaOmron':
        raise ValueError('Unsupported official data format')
    receipt=json.loads((root.parent/'download.complete.json').read_text())
    if receipt.get('task')!=entry['task'] or receipt.get('url')!=entry['url']:
        raise ValueError('Missing or mismatched official download receipt')
    if len(episodes)!=info['total_episodes']:
        raise ValueError(f'Episode count mismatch: {root}')
    if info.get('total_videos')!=len(episodes)*len(CAMERAS):raise ValueError('Video count mismatch')
    ids=[ep['episode_index'] for ep in episodes]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate episode IDs')
    if sum(ep['length'] for ep in episodes)!=info['total_frames']:
        raise ValueError('Frame count mismatch')
    for ep in episodes:
        fmt={'episode_chunk':ep['episode_index']//info['chunks_size'],'episode_index':ep['episode_index']}
        paths=[root/info['data_path'].format(**fmt)]
        paths.extend(root/info['video_path'].format(**fmt,video_key='observation.images.'+c) for c in CAMERAS)
        for path in paths:
            if not path.is_file() or path.stat().st_size==0:
                raise FileNotFoundError(f'Missing/empty official payload: {path}')
