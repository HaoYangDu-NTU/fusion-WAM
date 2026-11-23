"""Run preselected, paired gain interventions after the full benchmark sequence."""
from collections import defaultdict
from datetime import datetime
import fcntl
from pathlib import Path
import random
import sys
import time

import e00_full_benchmarks as full
import e00_router_sequence as seq

OUT = seq.FROZEN.parent/'residual_gain_20261121'


def draw(rows, seed):
    strata = defaultdict(list)
    for row in rows:
        strata[row['suite'], row['category']].append(row)
    rng = random.Random(seed)
    if len(strata) != 12 or any(len(cell) != 50 for cell in strata.values()):
        raise ValueError('Expected the existing 600-trial stratified design')
    return [r for key in sorted(strata) for r in rng.sample(sorted(strata[key], key=lambda x:x['name']), 10)]


def design():
    screen = draw(seq.selected(), 2026112101)
    confirm = draw(seq.read(full.causal.PARENT/'holdout_selection.json'), 2026112102)
    if {r['name'] for r in screen} & {r['name'] for r in confirm}:
        raise ValueError('Screen and confirmation overlap')
    locked = {'screen':screen, 'confirmation':confirm, 'gains':[0, 1, 2],
              'confirmation_rule':'per arm: repeat 0 vs 1 if abs(net)>=6/120; repeat 2 vs 1 '
                                  'if counts are strictly monotone across gains and abs(2 minus 1)>=6/120'}
    path = OUT/'design.json'
    if path.exists() and seq.read(path) != locked:
        raise ValueError('Task design changed on restart')
    seq.write(path, locked)
    return locked


def trials(arm, scale, rows, phase, checkpoint):
    deploy = full.OUT/f'seed42_{arm}'/'plus_deployment'/checkpoint.name
    if deploy.resolve(strict=True) != checkpoint.resolve(strict=True):
        raise ValueError(f'Plus deployment differs from full benchmark checkpoint: {deploy}')
    template = seq.read(seq.OLD/'decoder_only/command.json')
    template[1:3] = [str(seq.SCRIPTS/'e00_gain_eval.py'), 'evaluate', str(scale)]
    template[template.index('--base-port')+1] = '10620'
    label = f'{phase}/{arm}_{scale}x'
    result = seq.evaluate(label, deploy, rows, command_template=template)
    for folder in (OUT/label).glob('attempt_*/evaluation'):
        if not seq.trial_results(folder, set(result)):
            continue
        manifest = seq.read(folder/'run.json')
        if (Path(manifest['checkpoint_dir'])/manifest['checkpoint_name']).resolve() != checkpoint:
            raise ValueError(f'Incorrect scored checkpoint: {folder}')
        receipts = [seq.read(path) for path in folder.glob('gain_*.json')]
        if not receipts or any(r['scale'] != scale for r in receipts):
            raise ValueError(f'Missing intervention evidence: {folder}')
        for item in manifest['plan']['servers']:
            cmd = item['command']
            if cmd[1:4] != [str(seq.SCRIPTS/'e00_gain_eval.py'), 'server', str(scale)]:
                raise ValueError(f'Model replica used the wrong intervention: {folder}')
    return result


def contrast(ref, changed, rows):
    return {'rates':seq.compare(ref, changed, rows),
            'rescued':[n for n in ref if changed[n] > ref[n]],
            'lost':[n for n in ref if changed[n] < ref[n]]}


def confirmation_gains(counts):
    off, native, double = (counts[k] for k in (0, 1, 2))
    gains = []
    if abs(off-native) >= 6:
        gains.append(0)
    if (off < native < double or off > native > double) and abs(double-native) >= 6:
        gains.append(2)
    return gains


def report(data):
    lines = ['# M29 闭环评测与残差倍率分析','',
             '主结果使用标准 LIBERO、全量 LIBERO-Plus 及预留 600 条。倍率筛查与确认各 120 条，两个清单无重叠，均取自全量 Plus。','']
    for name, arms in data['full'].items():
        lines += [f'## {name} 最终权重','', '| 模型 | 成功 / 回合 | 成功率 |','|---|---:|---:|']
        for arm, stats in arms.items():
            s,n = stats['successes'], stats['trials']
            lines.append(f'| {arm} | {s}/{n} | {s/n:.2%} |')
        delta = arms['seed42_sparse']['successes'] - arms['seed42_router']['successes']
        lines += ['', f"seed 42 的 Sparse 相对 Dense 净变化 {delta:+d} 回合（{100*delta/arms['seed42_sparse']['trials']:+.2f} 个百分点）。",'']
    hold = data['holdout']
    lines += [f"预留 600 条上 Sparse 相对 Dense 救回 {hold['gained']} 条，新增失败 {hold['lost']} 条，"
              f"成功率变化 {hold['delta_pp']:+.2f} 个百分点。",'']
    lines += ['## 训练与固定输入的动作敏感性','',
              '| 模型 | 最后 100 步 Top-32 注意力占比 | 残差比例 | 移除 Top-32 动作 RMS 差 | 随机移除 32 个 | 等质量衰减 |',
              '|---|---:|---:|---:|---:|---:|']
    for arm, row in data['mechanism'].items():
        lines.append(f"| {arm} | {row['mass']:.1%} | {row['residual']:.3%} | "
                     f"{row['top32']:.5g} | {row['random32']:.5g} | {row['equal_mass']:.5g} |")
    lines.append('')
    for phase, title in [('screen','倍率筛查'), ('confirmation','独立变体确认')]:
        lines += [f'## {title}','','| 模型 | 倍率 | 1× 成功 | 实验成功 | 救回 | 新增失败 | 净变化（百分点） |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for arm, comps in data[phase].items():
            for gain, comp in comps.items():
                r = comp['rates']['overall']
                lines.append(f"| {arm} | {gain}× | {r['baseline']}/120 | {r['router']}/120 | "
                             f"{r['gained']} | {r['lost']} | {r['delta_pp']:+.2f} |")
        lines.append('')
    for arm, comparisons in data['confirmation'].items():
        if not comparisons:
            lines.append(f'{arm} 的筛查未触发确认阈值。该样本量仍可能漏掉较小的效应。')
        for gain, comp in comparisons.items():
            a = data['screen'][arm][gain]['rates']['overall']['delta_pp']
            b = comp['rates']['overall']['delta_pp']
            direction = '在两个清单上同向' if a*b > 0 else '未在确认清单重现同向变化'
            lines.append(f'{arm} 的 {gain}× 相对 1× {direction}，筛查和确认分别为 {a:+.2f}、{b:+.2f} 个百分点。')
    for arm, comp in data.get('against_baseline', {}).items():
        lines.append(f"{arm} 在同一筛查清单上相对单独训练的无 Router 基线，1× 净变化 "
                     f"{comp[1]['rates']['overall']['delta_pp']:+.2f} 个百分点，0× 净变化 "
                     f"{comp[0]['rates']['overall']['delta_pp']:+.2f} 个百分点。")
    if set(data['confirmation']) == {'router','sparse'}:
        fullplus = data['full']['libero-plus']
        sparse_win = hold['delta_pp'] > 0 and fullplus['seed42_sparse']['successes'] > fullplus['seed42_router']['successes']
        confirmed = data['confirmation']['sparse']
        if sparse_win and 0 in confirmed and confirmed[0]['rates']['overall']['delta_pp'] < 0 and data['screen']['sparse'][0]['rates']['overall']['delta_pp'] < 0:
            advice = 'Sparse 在预留集、全量 Plus 上占优，且关闭分支在两个倍率清单均降低成绩。优先用第二个训练种子复现 Sparse 对 Dense。'
        elif any(2 in comps and comps[2]['rates']['overall']['delta_pp'] > 0 and data['screen'][arm][2]['rates']['overall']['delta_pp'] > 0 for arm, comps in data['confirmation'].items()):
            advice = '放大分支在筛查与独立确认中均有增益。下一项训练只改变门控初始化，固定 Router 学习率和训练预算。'
        elif sparse_win:
            advice = 'Sparse 在两个成功率比较上占优。倍率实验尚不能将该收益归因于推理分支；优先复现跨训练种子的成功率差异。'
        elif any(all(comp[g]['rates']['overall']['delta_pp'] >= 5 for g in (0,1))
                 and fullplus['seed42_'+arm]['successes'] > fullplus['seed42_baseline']['successes']
                 for arm, comp in data.get('against_baseline', {}).items()):
            advice = '至少一个模型在全量 Plus 上优于无 Router 基线，筛查中关闭分支仍保留优势。下一项可做训练期梯度对照，检验分支训练对共享主干的影响。'
        else:
            advice = '预留集与全量 Plus 未共同显示 Sparse 优于 Dense。结合上表的动作敏感性和按类别计数判断，当前不优先扩展稀疏超参数搜索。'
        lines += ['', '## 后续实验判断','', advice]
    lines += ['', '倍率试验固定 seed 42 的权重，只改变推理时分支的残差强度。120 条的组别差异用于筛查。'
              '同一基础任务的变体可能相关。逐条救回、新增失败和类别计数见 results.json。',
              f'训练轨迹及 Top-32 干预见 [稀疏对照报告]({full.causal.OUT}/REPORT.md)。']
    seq.write(OUT/'results.json', data)
    tmp = OUT/'REPORT.tmp'
    tmp.write_text('\n'.join(lines)+'\n')
    tmp.replace(OUT/'REPORT.md')


def run():
    locked = design()
    while not (full.OUT/'DONE.json').is_file():
        if any((full.OUT/name).exists() for name in ('delivered.json','delivery_failed.json')):
            raise RuntimeError('Full benchmark queue stopped before completion')
        seq.status('waiting_for_full_benchmarks', upstream=seq.read(full.OUT/'status.json')['stage'])
        time.sleep(60)
    weights = seq.read(full.OUT/'checkpoints.json')
    causal = seq.read(full.causal.OUT/'results.json')
    mechanism = {}
    for arm in ('router','sparse'):
        last = causal['training'][arm]['10690']
        cond = causal['ablation'][arm]['conditions']
        mechanism[arm] = {'mass':last['router_topk_mass'], 'residual':last['router_residual_ratio'],
                          **{k:cond[k]['components']['all']['rms'] for k in ('top32','random32','equal_mass')}}
    data = {'full':seq.read(full.OUT/'results.json'),
            'holdout':causal['holdout']['sparse_minus_dense']['overall'], 'mechanism':mechanism,
            'screen':{}, 'confirmation':{}, 'against_baseline':{}}
    old_baseline = seq.read(full.causal.PARENT/'seed42/pilot_baseline/trials.json')
    for arm in ('router', 'sparse'):
        checkpoint = Path(weights['seed42_'+arm]).resolve(strict=True)
        rows = locked['screen']
        measurements = {g:trials(arm,g,rows,'screen',checkpoint) for g in (1,0,2)}
        baseline = {r['name']:old_baseline[r['name']] for r in rows}
        data['against_baseline'][arm] = {g:contrast(baseline,measurements[g],rows) for g in (0,1)}
        data['screen'][arm] = {g:contrast(measurements[1],measurements[g],rows) for g in (0,2)}
        follow = confirmation_gains({g:sum(v.values()) for g,v in measurements.items()})
        data['confirmation'][arm] = {}
        if follow:
            rows = locked['confirmation']
            native = trials(arm,1,rows,'confirmation',checkpoint)
            for g in follow:
                data['confirmation'][arm][g] = contrast(native,trials(arm,g,rows,'confirmation',checkpoint),rows)
        report(data)
    seq.write(OUT/'DONE.json', {'finished':datetime.now().isoformat()})
    seq.status('completed')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv)>1:
        from e00_sequence_supervise import main
        main(OUT, Path(__file__).name, 'M29 门控倍率、完整闭环结果与下一步实验分析')
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
