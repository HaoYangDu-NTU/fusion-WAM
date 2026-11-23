"""Run the full LIBERO benchmarks on the final E00 checkpoints after both queues."""
from datetime import datetime
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import yaml
import e00_causal_sequence as causal
import e00_router_sequence as seq

OUT = seq.FROZEN.parent / 'full_benchmarks_20261121'
BENCHMARKS = {'libero': 2000, 'libero-plus': 10030}


def wait_for_predecessors():
    while not (causal.OUT/'DONE.json').is_file():
        for parent in (causal.PARENT, causal.OUT):
            if not (parent/'DONE.json').is_file() and any(
                (parent/name).is_file() for name in ('delivered.json', 'delivery_failed.json')
            ):
                raise RuntimeError(f'Upstream queue stopped at {parent}: {seq.read(parent/"status.json")}')
        seq.status('waiting_for_training_and_queued_evaluations',
                   training_queue=seq.read(causal.PARENT/'status.json'),
                   sparse_queue=seq.read(causal.OUT/'status.json')['stage'])
        time.sleep(60)


def checkpoints():
    dirs = {f'seed42_{arm}': causal.PARENT/'seed42'/arm for arm in ('baseline', 'router')}
    dirs['seed42_sparse'] = causal.OUT/'sparse_train'
    if 'holdout43' in seq.read(causal.PARENT/'results.json'):
        dirs.update({f'seed43_{arm}': causal.PARENT/'seed43'/arm for arm in ('baseline', 'router')})
    return {arm: Path(seq.read(folder/'index.json')['10690']).resolve(strict=True)
            for arm, folder in dirs.items()}


def counts(summary, benchmark):
    if not summary['complete'] or summary['smoke']:
        raise ValueError(f'Incomplete evaluation for {benchmark}')
    if benchmark == 'libero':
        groups = {r['suite']: [r['successes'], r['valid_trials']] for r in summary['suites']}
    else:
        groups = {r['category']: [r['successes'], r['trials']] for r in summary['category_results']}
    total = sum(row[1] for row in groups.values())
    successes = sum(row[0] for row in groups.values())
    if total != BENCHMARKS[benchmark] or any(not 0 <= s <= n for s, n in groups.values()):
        raise ValueError(f'Invalid {benchmark} counts: {groups}')
    return {'successes': successes, 'trials': total, 'groups': groups}


def report(results):
    lines = ['# M29 最终权重闭环评测', '', '同一训练种子的各组均使用第 10,690 步权重。', '']
    for benchmark, arms in results.items():
        lines += [f'## {benchmark}', '', '| 模型 | 成功数 / 回合数 | 成功率 | 相对同种子基线 |',
                  '|---|---:|---:|---:|']
        for arm, row in arms.items():
            rate = row['successes']/row['trials']
            base = arms[arm.split('_')[0]+'_baseline']
            delta = 100 * (rate - base['successes']/base['trials'])
            lines.append(f"| {arm} | {row['successes']} / {row['trials']} | {rate:.2%} | {delta:+.2f} 个百分点 |")
        if 'seed42_sparse' in arms:
            diff = arms['seed42_sparse']['successes'] - arms['seed42_router']['successes']
            lines += ['', f"seed 42 下，Sparse 比 Dense 多成功 {diff:+d} 回合，"
                      f"差值为 {100*diff/BENCHMARKS[benchmark]:+.2f} 个百分点。"]
        lines += ['', '| 分组 | ' + ' | '.join(arms) + ' |', '|---|' + '---:|'*len(arms)]
        for name in next(iter(arms.values()))['groups']:
            cells = []
            for row in arms.values():
                success, n = row['groups'][name]
                cells.append(f'{success}/{n} ({success/n:.1%})')
            lines.append('| ' + ' | '.join([name, *cells]) + ' |')
        lines.append('')
    seq.write(OUT/'results.json', results)
    temp = OUT/'REPORT.tmp'
    temp.write_text('\n'.join(lines)+'\n')
    temp.replace(OUT/'REPORT.md')


def run():
    wait_for_predecessors()
    weights = checkpoints()
    seq.write(OUT/'checkpoints.json', {k: str(v) for k, v in weights.items()})
    cache = yaml.safe_load((seq.ROOT/'configs/protocol.yaml').read_text())['text']['cache_plus']
    results = {}
    for benchmark in BENCHMARKS:
        results[benchmark] = {}
        for arm, checkpoint in weights.items():
            job = OUT/arm/benchmark
            job.mkdir(parents=True, exist_ok=True)
            env = {**seq.ENV, 'TEXT_EMBEDDING_DIR': str(seq.ROOT/cache),
                   'PLUS_DEPLOY_DIR': str(OUT/arm/'plus_deployment'), 'OPENWAM_PYTHON': seq.PYTHON}
            cmd = [seq.PYTHON, '-u', '-m', 'benchmarks.vla_harness.run_resumable',
                   '--benchmarks', benchmark, '--ckpt-dir', str(checkpoint.parent), '--ckpt-name', checkpoint.name,
                   '--output-dir', str(job), '--gpus', '0,1,2,3', '--workers', '32', '--max-batch-size', '8',
                   '--render-gpus', '0,1,2,3', '--base-port', '10520', '--wait-for-idle',
                   '--correct-instruction-suffix', 'true']
            seq.status('full_evaluation', arm=arm, benchmark=benchmark, checkpoint=str(checkpoint),
                       progress_file=str(job/'status.json'))
            # The native runner validates checkpoint identity and recovers whole tasks on restart.
            with (job/'console.log').open('a') as log:
                subprocess.run(cmd, env=env, cwd=seq.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not (job/'DONE').exists():
                raise RuntimeError(f'Missing final evaluation receipt in {job}')
            results[benchmark][arm] = counts(seq.read(job/benchmark/'summary.json'), benchmark)
            report(results)
    seq.write(OUT/'DONE.json', {'finished': datetime.now().isoformat()})
    seq.status('completed')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        from e00_sequence_supervise import main
        main(OUT, Path(__file__).name, 'M29 最终权重完整 LIBERO 与 LIBERO-Plus 评测')
    else:
        seq.OUT = OUT
        with (OUT/'sequence.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if not (OUT/'DONE.json').exists():
                    run()
            except Exception as exc:
                seq.status('needs_attention', error=str(exc))
                raise
