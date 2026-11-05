"""Aggregate cumulative per-process evaluation counters without double counting."""

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize_performance(directory):
    roles = {}
    for path in sorted(Path(directory).glob('*.jsonl')):
        latest = None
        for line in path.read_text().splitlines():
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                # A live writer can have an unfinished final line.
                continue
        if latest is None:
            continue
        role = roles.setdefault(latest['role'], {'processes': 0, 'seconds': Counter(), 'counts': Counter()})
        role['processes'] += 1
        role['seconds'].update(latest['seconds'])
        role['counts'].update(latest['counts'])
    client = roles.get('client', {}).get('counts', {})
    requests = client.get('request_roundtrip', 0)
    model = roles.get('model', {}).get('counts', {})
    batches = {int(key.removeprefix('batch_size_')): count
               for key, count in model.items() if key.startswith('batch_size_')}
    batch_count = sum(batches.values())
    return {'roles': roles, 'control_actions_per_request': client.get('control_actions', 0) / requests if requests else None,
            'actual_batch_histogram': batches,
            'mean_actual_batch': sum(n * count for n, count in batches.items()) / batch_count if batch_count else None,
            'timing_note': 'Summed CPU wall durations; overlapping phases are not additive and are not GPU kernel times.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='An attempt directory containing performance/, or performance/ itself')
    args = parser.parse_args()
    directory = args.directory / 'performance' if (args.directory / 'performance').is_dir() else args.directory
    if not directory.is_dir():
        parser.error(f'No performance directory: {directory}')
    print(json.dumps(summarize_performance(directory), indent=2))


if __name__ == '__main__':
    main()
