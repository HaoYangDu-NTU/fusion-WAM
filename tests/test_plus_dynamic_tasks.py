from concurrent.futures import ProcessPoolExecutor
import sqlite3

import pytest

from benchmarks.vla_harness.task_queue import DynamicWorkItems, initialize_queue


def consume(args):
    path, items = args
    work = DynamicWorkItems(path, items)
    assert len(work) == len(items) and work[0] == items[0]
    return [task['task_id'] for _, task, _ in work]


def test_processes_claim_every_task_once(tmp_path):
    items = [(i, dict(suite='libero_goal', task_id=i), 0) for i in range(503)]
    path = tmp_path/'queue.sqlite'
    initialize_queue(path, [task for _, task, _ in items])
    with ProcessPoolExecutor(max_workers=8) as pool:
        groups = list(pool.map(consume, [(path, items)]*8))
    observed = [i for group in groups for i in group]
    assert len(observed) == 503 and sorted(observed) == list(range(503))


def test_fast_worker_can_take_slow_workers_pending_tasks(tmp_path):
    items = [(i, dict(suite='s', task_id=i), 0) for i in range(11)]
    path = tmp_path/'queue.sqlite'
    initialize_queue(path, [task for _, task, _ in items])
    slow = iter(DynamicWorkItems(path, items))
    first = next(slow)
    fast = list(DynamicWorkItems(path, items))
    assert len(fast) == 10 and first not in fast and list(slow) == []


def test_failed_attempt_requeues_uncommitted_claim(tmp_path):
    tasks = [dict(suite='s', task_id=i) for i in range(3)]
    path = tmp_path/'old.sqlite'
    initialize_queue(path, tasks)
    iterator = iter(DynamicWorkItems(path, [(i,t,0) for i,t in enumerate(tasks)]))
    unfinished = next(iterator)
    iterator.close()
    remaining = [task for task in tasks if task != tasks[(unfinished[0]+1)%3]]
    fresh = tmp_path/'new.sqlite'
    initialize_queue(fresh, remaining)
    retried = list(DynamicWorkItems(fresh, [(i,t,0) for i,t in enumerate(remaining)]))
    assert unfinished[1] in [t for _,t,_ in retried]


def test_rejects_nonzero_trials_and_duplicate_tasks(tmp_path):
    task = dict(suite='s', task_id=0)
    with pytest.raises(ValueError, match='Duplicate'):
        initialize_queue(tmp_path/'bad.sqlite', [task,task])
    with pytest.raises(ValueError, match='trial zero'):
        DynamicWorkItems(tmp_path/'missing', [(0,task,1)])
