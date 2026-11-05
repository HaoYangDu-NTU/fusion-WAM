"""Opt-in separation of GPU launch threads from DataLoader CPU work on Linux."""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _cpu_set(text):
    result = set()
    for item in text.strip().split(','):
        bounds = [int(n) for n in item.split('-')]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def training_cpu_plan():
    """Reserve two physical cores per local rank, respecting existing affinity."""
    if os.environ.get('OPENWAM_ISOLATE_TRAIN_CPUS') != '1':
        return None
    allowed = set(os.sched_getaffinity(0))
    groups = []
    remaining = set(allowed)
    while remaining:
        cpu = min(remaining)
        siblings = _cpu_set(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text()) & allowed
        groups.append(siblings)
        remaining -= siblings
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    if len(groups) <= 2 * world:
        raise ValueError('CPU isolation needs two physical cores per local rank plus DataLoader cores')
    reserved = set().union(*groups[:2 * world])
    training = set().union(*groups[2 * rank:2 * rank + 2])
    workers = allowed - reserved
    logger.info('CPU affinity rank %d: training=%s, workers=%d logical CPUs', rank, sorted(training), len(workers))
    return tuple(sorted(training)), tuple(sorted(workers))


def init_worker_affinity(worker_id, *, cpus, seed_fn=None):
    # Set this before PyAV creates decoder threads so they inherit the worker pool.
    os.sched_setaffinity(0, cpus)
    if seed_fn is not None:
        seed_fn(worker_id)


def pin_training_threads(cpus):
    """Pin existing CUDA/autograd/loader threads after first-step compilation.

    DataLoader workers are separate processes and keep their own affinity.
    Future threads inherit the training process's restricted CPU set.
    """
    for task in Path('/proc/self/task').iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except ProcessLookupError:
            pass  # A helper thread exited between enumeration and pinning.
