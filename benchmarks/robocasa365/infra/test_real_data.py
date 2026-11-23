"""Optional real-data integration; no downloads or simulator state replay.

ROBOCASA365_MANIFEST=<manifest> python -m pytest .../test_real_data.py -q
"""
import os
from pathlib import Path
import numpy as np
import pytest

MANIFEST=os.environ.get('ROBOCASA365_MANIFEST')
pytestmark=pytest.mark.skipif(not MANIFEST,reason='Set ROBOCASA365_MANIFEST for official real-data integration')


def test_original_values_boundaries_rgb_batch_and_native_dispatch():
    from .dataset import Human300Dataset,make_loader
    from .schema import CAMERAS,action_to_env
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
    ds=Human300Dataset(MANIFEST,history=3,chunk=4,tasks=['OpenDrawer'])
    first=ds[0];end=ds.ends[0]-1;last=ds[end];following=ds[end+1]
    assert first['observation']['history_valid'].tolist()==[False,False,True]
    assert last['action_valid'].tolist()==[True,False,False,False]
    assert first['episode_id']==last['episode_id']!=following['episode_id']
    assert following['frame_index']==0
    entry,info,ep=ds.episodes[0]
    fmt={'episode_chunk':ep['episode_index']//info['chunks_size'],'episode_index':ep['episode_index']}
    raw=ds._episode_table(str(Path(entry['path'])/info['data_path'].format(**fmt)))
    assert np.array_equal(first['observation']['state'][-1],raw['observation.state'][0])
    assert np.array_equal(first['action'],np.asarray(raw['action'])[:4])
    assert np.array_equal(first['action_timestamp'],np.asarray(raw['timestamp'])[:4])
    assert np.array_equal(last['action'],np.repeat(np.asarray(raw['action'])[-1:],4,axis=0))
    for camera in CAMERAS:
        image=first['observation']['rgb'][camera]
        assert image.shape==(3,256,256,3) and image.dtype==np.uint8
        assert image.max()>0 and image.std()>1
    native=first['action'][0]
    dispatch=PandaOmronKeyConverter.unmap_action(action_to_env(native))
    assert np.array_equal(dispatch['robot0_right'],native[5:11])
    assert np.array_equal(dispatch['robot0_base'],native[:3])
    assert np.array_equal(dispatch['robot0_torso'],native[3:4])
    batch=next(iter(make_loader(ds,batch_size=2,workers=2)))
    assert batch['observation']['state'].shape==(2,3,16)
    assert batch['action'].shape==(2,4,12)
    assert all(batch['observation']['rgb'][c].shape==(2,3,256,256,3) for c in CAMERAS)
