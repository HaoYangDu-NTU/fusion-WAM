"""Matched sparse routing, paired evaluation and value-path ablation for M29."""
import collections
from datetime import datetime
import fcntl
import json
from pathlib import Path
import sys
import time

import e00_router_sequence as seq

PARENT = seq.OUT
OUT = seq.FROZEN.parent / 'causalwam_followup_20261121'
SOURCE = 'https://arxiv.org/html/2610.42096v1'


def validate_trials(rows, tasks):
    if set(rows) != {r['name'] for r in tasks} or not all(type(v) is int and v in (0,1) for v in rows.values()):
        raise ValueError('Trial keys or score values disagree with the locked evaluation design')
    return rows


def score(arm, checkpoint, tasks, split):
    # Reuse the successful predecessor's exact per-trial export only for the
    # matching seed/arm, manifest and final checkpoint.
    old = PARENT/'seed42'/f'{split}_{arm}'
    if arm != 'sparse' and (old/'trials.json').is_file():
        attempts = list(old.glob('attempt_*/evaluation/run.json'))
        if not attempts:
            raise ValueError(f'Missing evaluation provenance in {old}')
        for manifest in attempts:
            m = seq.read(manifest)
            weights = Path(m['checkpoint_dir'])/m['checkpoint_name']
            if weights.resolve() != checkpoint.resolve():
                raise ValueError(f'Mismatch in {manifest}')
        return validate_trials(seq.read(old/'trials.json'),tasks)
    return validate_trials(seq.evaluate(f'{split}_{arm}',checkpoint,tasks),tasks)


def tabulate(result, title):
    lines = [f'## {title}','','| 稀疏对稠密 | 稠密成功 | 稀疏成功 | 回合 | 变化（百分点） | 救回 | 新增失败 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for key, row in result.items():
        lines.append('| {} | {baseline} | {router} | {trials} | {delta_pp:+.2f} | {gained} | {lost} |'.format(key, **row))
    return lines+['']


def training_snapshots(checkpoint):
    fields = ('router_gate', 'router_topk_mass', 'router_entropy', 'router_residual_ratio')
    windows = {step: {} for step in (1000, 5000, 10690)}
    with (checkpoint.parent/'router_metrics.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            for end, rows in windows.items():
                if end - 100 < row['step'] <= end:
                    rows[row['step']] = row
    snapshots = {}
    for end, rows in windows.items():
        if set(rows) != set(range(end-99, end+1)):
            raise ValueError(f'Incomplete training metrics through {end} for {checkpoint}')
        snapshots[end] = {key: sum(row[key] for row in rows.values())/100 for key in fields}
    return snapshots


def report(data):
    lines = ['# OpenWAM M29 稀疏 Router 对照','', '主指标为 seed 42、相同最终训练步下，稀疏组相对同容量稠密组在预留 600 个任务变体上的配对成功率差。','']
    for split,title in [('holdout','预留清单，主结果'),('pilot','历史 pilot 清单，次结果')]:
        if split not in data:
            continue
        d = data[split]
        lines.extend(tabulate(d['sparse_minus_dense'],title))
        lines += [f"该清单无 Router 基线成功 {d['sparse_minus_baseline']['overall']['baseline']}/600，"
                  f"稀疏组相对无 Router 基线变化 {d['sparse_minus_baseline']['overall']['delta_pp']:+.2f} 个百分点。",'']
    if 'training' in data:
        lines += ['## 训练中的门控与注意力','','各数值取截至标记步数的最后 100 次更新均值。','',
                  '| 模型 | 步数 | gate | Top-32 注意力占比 | 注意力熵 | 门控残差与输入范数比 |',
                  '|---|---:|---:|---:|---:|---:|']
        for arm, snapshots in data['training'].items():
            for step, row in snapshots.items():
                lines.append(f"| {arm} | {step} | {row['router_gate']:.5f} | {row['router_topk_mass']:.2%} | "
                             f"{row['router_entropy']:.3f} | {row['router_residual_ratio']:.3%} |")
        lines.append('')
    if 'ablation' in data:
        lines += ['## 固定输入的分支干预','','| 模型 | 干预 | 平均移除 attention mass | 动作相对 L2 | 平移动作 RMS 差 | 旋转动作 RMS 差 | 夹爪 RMS 差 |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for arm, res in data['ablation'].items():
            for mode in ('top32','random32','equal_mass'):
                row=res['conditions'][mode]
                c=row['components']
                relative=c['all']['relative_l2']
                lines.append(f"| {arm} | {mode} | {row['mean_removed_attention_mass']:.4f} | "
                             f"{relative if relative is None else format(relative,'.5g')} | "
                             f"{c['translation']['rms']:.5g} | {c['rotation']['rms']:.5g} | {c['gripper']['rms']:.5g} |")
        lines += ['','此诊断使用四个预定模拟初态，在同一观测和噪声上干预末端分支。主干仍能看到视觉上下文。'
                  '等质量对照校准移除的注意力总量，value 向量的方向和范数仍可不同。','']
    if 'holdout' in data:
        delta=data['holdout']['sparse_minus_dense']['overall']['delta_pp']
        lines += [f'在这个训练种子和预留集合上，稀疏正则带来的净变化为 {delta:+.2f} 个百分点。'
                  '跨训练种子的可重复性仍待验证。','']
    lines += [f'实验依据见 [CausalWAM 正文及附录 H]({SOURCE})。']
    seq.write(OUT/'results.json',data)
    tmp=OUT/'REPORT.tmp'
    tmp.write_text('\n'.join(lines)+'\n')
    tmp.replace(OUT/'REPORT.md')


def run():
    pilot=seq.selected()
    holdout=seq.read(PARENT/'holdout_selection.json')
    if {r['name'] for r in pilot} & {r['name'] for r in holdout}:
        raise ValueError('Pilot and holdout overlap')
    for rows in (pilot,holdout):
        cells=collections.Counter((r['category'],r['suite']) for r in rows)
        if len({r['name'] for r in rows})!=600 or len(cells)!=12 or set(cells.values())!={50}:
            raise ValueError('Unexpected evaluation strata')
    while not (PARENT/'DONE.json').is_file():
        state=seq.read(PARENT/'status.json')
        seq.status('waiting_for_prior_sequence',prior=state)
        if state['stage']=='needs_attention' and any((PARENT/f).exists() for f in ('delivered.json','delivery_failed.json')):
            raise RuntimeError(f'Predecessor stopped before completion; inspect {PARENT}/status.json')
        time.sleep(60)
    matched=PARENT/'seed42'
    weights={arm:Path(seq.read(matched/arm/'index.json')['10690']) for arm in ('baseline','router')}
    sparse=seq.component('e00_causal_train.py',['--matched-root',str(matched)],OUT/'sparse_train','index.json')
    weights['sparse']=Path(sparse['10690'])
    data={'checkpoints':{k:str(v) for k,v in weights.items()},
          'training':{arm:training_snapshots(weights[arm]) for arm in ('router','sparse')}}
    report(data)
    for split,tasks in [('holdout',holdout),('pilot',pilot)]:
        rows={arm:score(arm,path,tasks,split) for arm,path in weights.items()}
        data[split]={k:seq.compare(rows[ref],rows['sparse'],tasks) for k,ref in
                     [('sparse_minus_dense','router'),('sparse_minus_baseline','baseline')]}
        report(data)
    data['ablation']={arm:seq.component('e00_causal_ablate.py',['--checkpoint',str(weights[arm])],
                                        OUT/f'ablate_{arm}','result.json') for arm in ('router','sparse')}
    report(data)
    seq.write(OUT/'DONE.json',{'finished':datetime.now().isoformat()})
    seq.status('completed')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1:
        from e00_sequence_supervise import main
        main(OUT,Path(__file__).name,'CausalWAM × M29 稀疏路由对照与干预')
    else:
        seq.OUT=OUT
        with (OUT/'sequence.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                if not (OUT/'DONE.json').exists():
                    run()
            except Exception as exc:
                seq.status('needs_attention', error=str(exc))
                raise
