"""Cache frozen E06 independent image features once per actual LIBERO timestamp."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from openwam.experiments import resolve_experiment
from openwam.dataloader.registry import build_dataset
from openwam.dataloader.transforms.multiview import assemble_two_view_layout
from openwam.model.video_backbone.encoder.flux2_framewise import Flux2FramewiseEncoder


def atomic_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n');temporary.replace(path)


class CurrentStereoFrames(Dataset):
    """Use the unchanged reader's camera decode/layout, without decoding future windows."""
    def __init__(self, reader):
        self.reader = reader
        assert reader._window_stride == 1 and reader._seperate_encode_view
        assert reader._color_jitter is None and reader._head_camera_choices is None
        assert np.all(reader._ep_valid_start == 0), 'Trimmed episodes need explicit historical-frame coverage'
        assert len(reader) == int(reader._eps_df['length'].sum())

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        ds = self.reader
        episode = int(np.searchsorted(ds._cum_n_starts, index, side='right')-1)
        offset = int(ds._ep_valid_start[episode]) + index-int(ds._cum_n_starts[episode])
        row = ds._eps_df.iloc[episode]
        local = np.array([0], dtype=np.int64)
        head = ds._decode_one_camera(ds._head_camera, row, episode, offset, local, kind='head')
        wrist = ds._decode_one_camera(ds._left_wrist_camera, row, episode, offset, local, kind='wrist')
        if not head:
            raise RuntimeError(f'Empty head frame at {index}')
        layout = getattr(ds, 'assemble_stereo', assemble_two_view_layout)
        image = layout(head[0], wrist[0] if wrist else None)
        return dict(index=index, episode=int(row['episode_index']), frame=offset,
                    pixels=torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2,0,1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit-frames', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--loader-workers', type=int, default=2)
    parser.add_argument('--write-inflight', type=int, default=8)
    parser.add_argument('--layout-only', action='store_true')
    parser.add_argument('--cache-dir', help='Separate cache directory for bounded validation probes')
    args = parser.parse_args()
    torch.set_num_threads(1)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    cfg, profile, _=resolve_experiment('E06')
    reader=build_dataset(cfg.dataloader, split='train');frames=CurrentStereoFrames(reader)
    layout_checks=[]
    for index in [0,64,68,len(reader)-1]:
        current=frames[index]['pixels'].permute(1,2,0).numpy()
        full=np.array(reader._getitem_impl(index)['video'][0].convert('RGB'))
        assert np.array_equal(current,full), f'Single frame/full window preprocessing differs: {index}'
        layout_checks.append(index)
    if args.layout_only:
        print(json.dumps(dict(passed=True,windows=len(reader),indices=layout_checks)),flush=True);return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU cache generation requires CUDA')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1:torch.distributed.init_process_group('gloo')
    config=dict(cfg.model.video_backbone.encoder);config.pop('name')
    if args.cache_dir:config['feature_cache_dir']=args.cache_dir
    cache_dir=Path(config['feature_cache_dir'])
    encoder=Flux2FramewiseEncoder.from_pretrained(**config).to('cuda')
    assert not any(p.requires_grad for p in encoder.parameters())
    cache=encoder.feature_cache
    assert cache is not None
    identity=dict(dataloader=OmegaConf.to_container(cfg.dataloader,resolve=True),windows=len(frames),
                  feature_cache_identity=cache.identity,layout_checks=layout_checks)
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    stop=min(len(frames),args.limit_frames or len(frames))
    assigned=list(range(rank,stop,world))
    report_path=cache_dir/f'build_rank{rank}.json'
    index_path=cache_dir/f'index_rank{rank}.jsonl'
    previous=json.loads(report_path.read_text()) if report_path.exists() else None
    reusable={}
    if index_path.exists():
        if previous is None or previous.get('identity_hash')!=identity_hash:
            raise RuntimeError('Cannot reuse a frame index without matching dataset and encoder identity')
        assigned_set=set(assigned)
        for line in index_path.read_text().splitlines():
            try:record=json.loads(line)
            except json.JSONDecodeError:continue  # interrupted final JSONL write
            index=record['index'];key=record['key']
            if index in assigned_set and (cache.directory/key[:2]/(key+'.safetensors')).exists():
                reusable[index]=record
    indices=[index for index in assigned if index not in reusable]
    loader=DataLoader(frames,batch_size=args.batch_size,sampler=indices,num_workers=args.loader_workers,
                      pin_memory=True,**(dict(prefetch_factor=2) if args.loader_workers else {}))
    started=time.monotonic();reused=len(reusable);processed=reused;misses=0;hits=reused;pending=[]
    verified=previous.get('verified',[]) if previous and previous.get('identity_hash')==identity_hash else []
    def progress():
        elapsed=time.monotonic()-started
        fresh=processed-reused
        return dict(rank=rank,world_size=world,processed=processed,expected=len(assigned),
            total_frames=len(frames),full_cache=False,elapsed_seconds=elapsed,
            frames_per_second=fresh/max(elapsed,1e-6),reused_index_frames=reused,
            estimated_remaining_seconds=(len(assigned)-processed)*elapsed/max(fresh,1),
            hits=hits,misses=misses,verified=verified,identity_hash=identity_hash,
            cache_namespace=str(cache.directory),gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
    def write(pair_cpu, features_cpu):
        return cache.get_or_encode(pair_cpu,lambda _:features_cpu)
    with ThreadPoolExecutor(max_workers=2) as writer, index_path.open('w') as index_file:
        for index in sorted(reusable):index_file.write(json.dumps(reusable[index])+'\n')
        index_file.flush()
        atomic_json(report_path,progress())
        print('CACHE_RESUME '+json.dumps(dict(rank=rank,reused=reused,remaining=len(indices))),flush=True)
        for batch in loader:
            pixels=batch['pixels'].to('cuda',torch.bfloat16,non_blocking=True).mul_(2.0/255.0).sub_(1)
            pairs=pixels.reshape(-1,3,224,2,224).permute(0,3,1,2,4).contiguous()
            cpu_pairs=pairs.cpu()
            # All images are independent; batch eight stereo timestamps per CUDA call.
            with torch.no_grad():
                batch_features=encoder._encode_pairs(pairs).cpu().contiguous()
            for j in range(len(pairs)):
                cpu_pair=cpu_pairs[j:j+1];key=cache.key(cpu_pair)
                path=cache.directory/key[:2]/(key+'.safetensors')
                with torch.no_grad():
                    if path.exists():
                        hits+=1
                    else:
                        z=batch_features[j:j+1]
                        assert z.shape==(1,2,128,14,14) and torch.isfinite(z).all()
                        pending.append(writer.submit(write,cpu_pair,z));misses+=1
                        if len(pending)>=args.write_inflight: pending.pop(0).result()
                    if len(verified)<4:
                        for future in pending: future.result()
                        pending.clear()
                        online=encoder._encode_pairs(pairs[j:j+1]).cpu()
                        cached=cache.get_or_encode(cpu_pair,lambda _: (_ for _ in ()).throw(RuntimeError('Missing cache')))
                        torch.testing.assert_close(online,cached,rtol=0,atol=0)
                        verified.append(dict(index=int(batch['index'][j]),bitwise_equal=True))
                index_file.write(json.dumps(dict(index=int(batch['index'][j]),episode=int(batch['episode'][j]),
                                                 frame=int(batch['frame'][j]),key=key))+'\n')
                processed+=1
            if (processed-reused) % 128==0 or processed==len(assigned):
                report=progress()
                atomic_json(report_path,report);index_file.flush()
                print(json.dumps(report),flush=True)
        for future in pending:future.result()
    assert processed==len(assigned)
    report=progress()
    report['local_writes_complete']=True
    atomic_json(report_path,report)
    # A window uses timestamps owned by multiple ranks. Every writer must finish
    # before any rank validates a whole window, especially at the episode tail.
    if world>1:torch.distributed.barrier()
    window_verified=[]
    if stop==len(frames) or stop>=64:
        window_indices=[64+rank,len(frames)-1-rank] if stop==len(frames) else [rank*4]
        for index in window_indices:
            sample=reader._getitem_impl(index)
            video=encoder.preprocess_video([sample['clean_condition_images'][0],*sample['video']])
            before_misses=cache.misses
            with torch.no_grad():
                cached=encoder.batch_encode(video)
                encoder.feature_cache=None
                try:online=encoder.batch_encode(video)
                finally:encoder.feature_cache=cache
            assert cached.shape==(1,128,10,14,28)
            assert cache.misses==before_misses, f'Window {index} was not completely precomputed'
            torch.testing.assert_close(cached,online,rtol=0,atol=0)
            window_verified.append(dict(index=index,bitwise_equal=True,all_cache_hits=True))
    report.update(full_cache=stop==len(frames),complete=True,identity=identity,
                  window_verified=window_verified,elapsed_seconds=time.monotonic()-started)
    atomic_json(report_path,report)
    print('CACHE_RANK_COMPLETE '+json.dumps(dict(rank=rank,frames=processed,full_cache=report['full_cache'])),flush=True)
    if world>1:torch.distributed.destroy_process_group()


if __name__=='__main__':main()
