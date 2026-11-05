"""CPU placement must preserve sampling, worker RNGs and epoch variation."""
from functools import partial
from pathlib import Path
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.train.utils import cpu_affinity as affinity
from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator


def test_disabled_plan_does_not_inspect_os(monkeypatch):
    monkeypatch.delenv('OPENWAM_ISOLATE_TRAIN_CPUS', raising=False)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: pytest.fail('disabled affinity inspected OS'))
    assert affinity.training_cpu_plan() is None


def test_four_rank_plan_keeps_siblings_together_and_respects_cpuset(monkeypatch):
    allowed = set(range(48, 96)) | set(range(144, 192))
    monkeypatch.setenv('OPENWAM_ISOLATE_TRAIN_CPUS', '1')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '4')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: allowed)
    monkeypatch.setattr(Path, 'read_text', lambda p: f'{int(p.parts[-3][3:]) % 96},{int(p.parts[-3][3:]) % 96 + 96}\n')
    for rank in range(4):
        monkeypatch.setenv('LOCAL_RANK', str(rank))
        training, workers = affinity.training_cpu_plan()
        assert set(training) == {48 + rank * 2, 49 + rank * 2, 144 + rank * 2, 145 + rank * 2}
        assert set(workers) == set(range(56, 96)) | set(range(152, 192))
        assert set(training).isdisjoint(workers)


def test_pin_current_process_threads_handles_exit_race(monkeypatch):
    monkeypatch.setattr(Path, 'iterdir', lambda _: iter([Path('/proc/self/task/10'), Path('/proc/self/task/11')]))
    seen = []
    def set_affinity(tid, cpus):
        seen.append((tid, cpus))
        if tid == 11:
            raise ProcessLookupError
    monkeypatch.setattr(os, 'sched_setaffinity', set_affinity)
    affinity.pin_training_threads((48, 144))
    assert seen == [(10, (48, 144)), (11, (48, 144))]


@pytest.mark.parametrize('seed', [None, 42])
def test_trainer_preserves_existing_worker_seed_callback(monkeypatch, seed):
    from openwam.train.openwam_trainer import OpenWAMTrainer
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.dataset = list(range(8))
    trainer.cfg = SimpleNamespace(training=SimpleNamespace(dataset_num_workers=2))
    trainer._run_seed, trainer._rank = seed, 0
    monkeypatch.setattr(affinity, 'training_cpu_plan', lambda: ((48, 144), (56, 152)))
    loader = trainer.build_dataloader(2)
    assert trainer._training_cpu_affinity == (48, 144)
    assert loader.worker_init_fn.func is affinity.init_worker_affinity
    assert loader.worker_init_fn.keywords == {
        'cpus': (56, 152), 'seed_fn': dataloader_worker_init_fn if seed is not None else None,
    }
    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)


class _RandomDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return (index, random.random(), float(np.random.random()), float(torch.rand(())))


def test_affinity_keeps_samples_and_random_values_identical_across_epochs():
    cpus = tuple(sorted(os.sched_getaffinity(0))[:2])
    def collect(worker_init):
        loader = torch.utils.data.DataLoader(
            _RandomDataset(), batch_size=4, num_workers=2, shuffle=True, collate_fn=list,
            generator=make_dataloader_generator(42, rank=0), worker_init_fn=worker_init,
        )
        return [[row for batch in loader for row in batch] for _ in range(2)]
    before = collect(dataloader_worker_init_fn)
    after = collect(partial(affinity.init_worker_affinity, cpus=cpus, seed_fn=dataloader_worker_init_fn))
    assert after == before
    assert after[0] != after[1]
