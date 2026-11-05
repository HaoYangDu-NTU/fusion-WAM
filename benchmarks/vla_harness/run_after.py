"""Wait for a completed evaluation and then execute the supplied evaluation command."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time


def evaluation_complete(directory):
    if not (directory / 'DONE').is_file():
        return False
    for benchmark in ('libero', 'libero-plus'):
        path = directory / benchmark / 'summary.json'
        if not path.is_file():
            return False
        summary = json.loads(path.read_text())
        if summary.get('smoke') or summary.get('selected_run_complete') is not True or summary.get('complete') is not True:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-output-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ready-file', type=Path, required=True, help='Completed text-cache manifest')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('An evaluation command is required after --')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (output / 'queue.json').write_text(json.dumps({
            'after_output_dir': str(args.after_output_dir.resolve()),
            'ready_file': str(args.ready_file.resolve()), 'command': command, 'pid': os.getpid()}, indent=2))
        print('Waiting for predecessor DONE, both complete summaries, and text cache.', flush=True)
        while not (output / 'STOP').exists() and not (output / 'DONE').exists():
            ready = args.ready_file.is_file() and json.loads(args.ready_file.read_text()).get('complete') is True
            if evaluation_complete(args.after_output_dir) and ready:
                print('Dependencies complete; starting evaluation.', flush=True)
                os.execvp(command[0], command)
            time.sleep(30)
        print('Queue stopped: STOP or DONE exists.', flush=True)


if __name__ == '__main__':
    main()
