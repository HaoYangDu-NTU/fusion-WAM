"""Run the three E00 router experiments serially; preserve trial- and stage-level evidence."""
import collections
from contextlib import closing
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openwam.experiments import runtime_env

FROZEN = ROOT / 'reports/E00_plus/frozen_router_20261111'
OLD = FROZEN / 'resume_20261121'
OUT = FROZEN / 'sequence_20261121'
SOURCE = FROZEN.parent / 'diagnosis_20261111'
SCRIPTS = ROOT / 'scripts'
PYTHON = '/data/fcy/basics/miniconda3/envs/openwam/bin/python'
ENV = {**os.environ, **runtime_env('0,1,2,3'), 'OPENBLAS_NUM_THREADS': '1'}
CATEGORIES = ['Camera Viewpoints', 'Light Conditions', 'Background Textures']


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def status(stage, **details):
    write(OUT / 'status.json', dict(stage=stage, updated=datetime.now().isoformat(), **details))


def selected():
    rows = read(SOURCE / 'plus_selection.json')['tasks']
    assert len(rows) == len({r['name'] for r in rows}) == 600
    return rows


def catalog():
    return [r for r in read(OLD / 'decoder_only/evaluation/task_catalog.json') if r['category'] in CATEGORIES]


def holdout():
    path = OUT / 'holdout_selection.json'
    if not path.exists():
        used = {r['name'] for r in selected()}
        strata = collections.defaultdict(list)
        for row in catalog():
            if row['name'] not in used:
                strata[row['suite'], row['category']].append(row)
        rng = random.Random(20261121)
        assert len(strata) == 12
        write(path, [r for key in sorted(strata) for r in rng.sample(sorted(strata[key], key=lambda x:x['name']), 50)])
    return read(path)


def trial_results(directory, tasks):
    dbs = list(directory.glob('recording-*.sqlite'))
    if not dbs or not dbs[0].is_file():
        return {}
    if len(dbs) != 1:
        raise ValueError(f'Ambiguous recording in {directory}')
    with closing(sqlite3.connect(dbs[0].resolve().as_uri()+'?mode=ro', uri=True, timeout=10)) as con:
        rows = con.execute('select task_name, episode_id, status, failure_reason from episode_results').fetchall()
    valid = {}
    for name, episode, outcome, error in rows:
        if name not in tasks or episode != 0:
            raise ValueError(f'Unexpected trial {name} in {directory}')
        if error or outcome not in ('success', 'fail'):
            continue
        if name in valid:
            raise ValueError(f'Duplicate trial {name}')
        valid[name] = int(outcome == 'success')
    return valid


def next_attempt(folder):
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / f'attempt_{1+len(list(folder.glob("attempt_*"))):03d}'
    result.mkdir()
    return result


def wait_for_cards():
    from benchmarks.vla_harness.run_resumable import resources_idle
    while not resources_idle('0,1,2,3', ()):
        time.sleep(30)


def execute(cmd, folder, label, watchdog=None):
    write(folder / 'command.json', cmd)
    logfile = folder / 'console.log'
    with logfile.open('w') as log:
        child = subprocess.Popen(cmd, env=ENV, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        status(label, pid=child.pid, directory=str(folder))
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                newest = logfile.stat().st_mtime
                if 'e00_sequence_probe.py' in Path(cmd[1]).name:
                    newest = max([newest, *(p.stat().st_mtime for p in Path(cmd[-1]).glob('*/*/evaluation/*.log'))])
                silent = time.time() - newest > 7200
                if (watchdog and watchdog()) or (not watchdog and silent):
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        code = child.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        code = child.wait()
                    break
    write(folder / 'exit.json', {'code':code, 'time':datetime.now().isoformat()})
    return code


def evaluate(label, checkpoint, tasks, previous=None, command_template=None):
    folder = OUT / label
    expected = {r['name'] for r in tasks}
    valid = {}
    def absorb(directory):
        for key, value in trial_results(directory, expected).items():
            if key in valid:
                raise RuntimeError(f'Duplicate scored task across attempts: {key}')
            valid[key] = value
    if previous:
        absorb(previous)
    for path in sorted(folder.glob('attempt_*/evaluation')):
        absorb(path)
    for retry in range(3):
        if valid.keys() == expected:
            break
        wait_for_cards()
        attempt = next_attempt(folder)
        remaining = expected - valid.keys()
        excluded = [[r['suite'], r['task_id']] for r in catalog() if r['name'] not in remaining]
        write(attempt / 'exclude.json', excluded)
        cmd = list(command_template) if command_template is not None else read(OLD / 'decoder_only/command.json')
        for flag, val in {'--ckpt-dir':checkpoint.parent, '--ckpt-name':checkpoint.name,
                          '--output-dir':attempt/'evaluation','--exclude-tasks-file':attempt/'exclude.json'}.items():
            cmd[cmd.index(flag)+1] = str(val)
        evaluated = attempt / 'evaluation'
        last_change, count = time.monotonic(), 0
        def stalled():
            nonlocal last_change, count
            current = len(trial_results(evaluated, remaining))
            if current > count:
                last_change, count = time.monotonic(), current
            return time.monotonic()-last_change > 2400
        execute(cmd, attempt, label, stalled)
        absorb(evaluated)
    if valid.keys() != expected:
        raise RuntimeError(f'{label} recovered {len(valid)} of {len(expected)} trials')
    write(folder / 'trials.json', valid)
    return valid


def compare(baseline, router, tasks):
    if set(baseline) != set(router) or set(router) != {r['name'] for r in tasks}:
        raise ValueError('Unmatched trial keys')
    result = {}
    for cat in ['overall', *CATEGORIES]:
        names = [r['name'] for r in tasks if cat == 'overall' or r['category']==cat]
        b, r = sum(baseline[n] for n in names), sum(router[n] for n in names)
        result[cat] = dict(trials=len(names), baseline=b, router=r, delta_pp=100*(r-b)/len(names),
                           gained=sum(router[n]>baseline[n] for n in names),
                           lost=sum(router[n]<baseline[n] for n in names))
    return result


def adopted_pilot():
    # The old launcher uses full-benchmark "complete" for a subset. SQLite trial keys
    # are authoritative, including when that old status becomes "failed" at 600/600.
    progress_time = time.monotonic()
    last = None
    while True:
        old_status = read(OLD/'status.json')
        if old_status['state'] != 'running':
            break
        alive = subprocess.run(['tmux','has-session','-t','=e00_frozen_pilot_1121'],
                               stderr=subprocess.DEVNULL).returncode == 0
        if not alive:
            break
        arm = old_status['arm']
        directory = OLD/arm/'evaluation'
        matched = trial_results(directory, {r['name'] for r in selected()})
        marker = arm, len(matched)
        if marker != last:
            progress_time, last = time.monotonic(), marker
            status('adopting_live_pilot', arm=arm, valid=len(matched), expected=600)
        if time.monotonic()-progress_time > 2400:
            pid = old_status['pid']
            cmdline = Path(f'/proc/{pid}/cmdline')
            if cmdline.exists():
                args = cmdline.read_bytes().decode().split('\0')
                if 'benchmarks.vla_harness.run_eval' in args and str(directory) in args:
                    os.kill(pid, signal.SIGTERM)
            # Leave the launching process to close its own workers and persist SQLite.
        time.sleep(30)
    groups = []
    for arm in ('decoder_only','fp32_dense'):
        ckpt, = (ROOT/'outputs/e00_plus_frozen_router/train'/arm).glob('*/checkpoint_step_1000.safetensors')
        groups.append(evaluate('pilot_'+arm, ckpt, selected(), OLD/arm/'evaluation'))
    return compare(*groups, selected())


def component(script, cmd, folder, result):
    artifact = folder / result
    if not artifact.is_file():
        for _ in range(2):
            wait_for_cards()
            attempt = next_attempt(folder/'logs')
            code = execute([PYTHON,str(SCRIPTS/script),*cmd,'--output',str(folder)], attempt, folder.name)
            if code == 0 and artifact.is_file():
                break
        else:
            raise RuntimeError(f'{script} failed for {folder}; see {attempt}')
    return read(artifact)


def probe(checkpoint, name, actions_only=False):
    flags = ['--actions-only'] if actions_only else []
    return component('e00_sequence_probe.py', ['--checkpoint',str(checkpoint),*flags], OUT/name, 'result.json')


def scratch(seed):
    checkpoints = {}
    for arm in ('baseline','router'):
        folder = OUT / f'seed{seed}' / arm
        index = component('e00_sequence_train.py',['--arm',arm,'--seed',str(seed)],folder,'index.json')
        checkpoints[arm] = Path(index['10690'])
        if arm == 'router' and seed == 42:
            for step in (1000,5500,10690):
                probe(Path(index[str(step)]), f'seed42/probe_{step}', actions_only=True)
    return checkpoints


def score_pair(checkpoints, seed, selection, name):
    scores = [evaluate(f'seed{seed}/{name}_{arm}', checkpoints[arm], selection) for arm in ('baseline','router')]
    return compare(*scores,selection)


def markdown(data):
    lines = ['# E00 门控残差实验结果','']
    for key, title in [('frozen','冻结主干，各 600 轮'),('scratch42','从头训练，seed 42'),
                       ('holdout42','预留任务，seed 42'),('holdout43','预留任务，seed 43')]:
        if key not in data:
            continue
        lines += [f'## {title}','','| 类别 | 基线成功 | Router 成功 | 总数 | 净变化（百分点） | 救回 | 新增失败 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for category, row in data[key].items():
            lines.append('| {} | {baseline} | {router} | {trials} | {delta_pp:+.2f} | {gained} | {lost} |'.format(category, **row))
        lines.append('')
    if 'on_off' in data:
        probe_data = data['on_off']
        loop = probe_data['closed_loop']
        actions = probe_data['paired_actions']
        comp = actions['components']
        lines += ['## 同权重 ON/OFF','',
                  f"标准 LIBERO 的四个固定 episode 中，开启 Router 成功 {loop['on']['successes']}/{loop['on']['trials']}，"
                  f"关闭成功 {loop['off']['successes']}/{loop['off']['trials']}。",
                  f"在相同观测、去噪噪声下测得 {actions['chunks']} 对动作块。平移、旋转、夹爪的原始动作 RMS 差依次为 "
                  f"{comp['translation']['rms']:.6g}、{comp['rotation']['rms']:.6g}、{comp['gripper']['rms']:.6g}。"
                  f"夹爪开合方向改变比例为 {comp['gripper']['sign_disagreement_fraction']:.1%}。",
                  '逐任务结果见 [探针记录](probe_frozen/result.json)。','']
    if 'confirmation' in data:
        lines += [data['confirmation'],'']
    if 'scratch42' in data:
        lines += ['## 从头训练中的门控贡献','',
                  '| 训练步 | 有效 gate | 实测残差比例 | ON/OFF 全动作 RMS 差 |',
                  '|---:|---:|---:|---:|']
        for step in (1000,5500,10690):
            result = read(OUT/f'seed42/probe_{step}/result.json')
            actions = result['paired_actions']
            ratio = actions['mean_over_observed_router_calls']['effective_residual_ratio']
            lines.append(f"| {step} | {result['gate']:.6g} | {ratio:.3%} | {actions['components']['all']['rms']:.6g} |")
        lines += ['', '主比较使用最终 10,690 步权重。600 个 pilot 任务在各组间逐条配对。'
                  '第一训练种子为 42。预留任务来自相同三个扰动类别，与 pilot 的任务变体不重叠。','']
    tmp = OUT/'REPORT.tmp'
    tmp.write_text('\n'.join(lines))
    tmp.replace(OUT/'REPORT.md')
    write(OUT/'results.json',data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT/'DONE.json').exists():
            return
        extra = holdout()
        data = {}
        status('waiting_for_existing_pilot')
        data['frozen'] = adopted_pilot()
        markdown(data)
        ckpt, = (ROOT/'outputs/e00_plus_frozen_router/train/fp32_dense').glob('*/checkpoint_step_1000.safetensors')
        data['on_off'] = probe(ckpt,'probe_frozen')
        markdown(data)
        models = scratch(42)
        data['scratch42'] = score_pair(models,42,selected(),'pilot')
        # Practical follow-up rule fixed before viewing either scratch outcome.
        if data['scratch42']['overall']['delta_pp'] >= 2:
            data['holdout42'] = score_pair(models,42,extra,'holdout')
            models43 = scratch(43)
            data['holdout43'] = score_pair(models43,43,extra,'holdout')
            data['confirmation'] = '首个训练种子在 pilot 净提升达到 2 个百分点，已在预留清单和第二个训练种子上复核。'
        else:
            data['confirmation'] = '首个训练种子在 pilot 的净提升低于预设的 2 个百分点，结束本轮，不触发额外确认训练。'
        markdown(data)
        write(OUT/'DONE.json',{'finished':datetime.now().isoformat()})
        status('completed')


if __name__=='__main__':
    try:
        main()
    except BlockingIOError:
        raise
    except Exception as error:
        status('needs_attention',error=str(error))
        raise
