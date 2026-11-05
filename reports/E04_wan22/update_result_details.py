"""Render complete suite/category results from source summaries into result.md."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
QUEUE = Path(__file__).resolve().parent / 'automatic_evaluation'
SOURCES = {
    'E01': ROOT / 'outputs/m29_rynn9b_dinov3/eval/s42/2026-10-24_02-30-16',
    'E02': ROOT / 'outputs/m29_rynn9b_ijepa/eval/s42/2026-10-22_23-15-46',
    'E03': ROOT / 'reports/E03_speed_optimization/automatic_evaluation/evaluation',
    'E04': QUEUE / 'evaluation',
}
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
CATEGORIES = ['Objects Layout', 'Camera Viewpoints', 'Robot Initial States',
              'Language Instructions', 'Light Conditions', 'Background Textures', 'Sensor Noise']


def load_results():
    results = {}
    for experiment, directory in SOURCES.items():
        for benchmark in ('libero', 'libero-plus'):
            path = directory / benchmark / ('evaluation/summary.json'
                if experiment == 'E02' and benchmark == 'libero-plus' else 'summary.json')
            data = json.loads(path.read_text())
            if not data.get('complete') or data.get('smoke'):
                results[experiment, benchmark] = None
                continue
            assert not data.get('invalid_records', 0), path
            if benchmark == 'libero':
                rows = data['suites']
                assert {r['suite'] for r in rows} == set(SUITES), path
                assert all(r['complete'] and not r['errors'] and r['valid_trials'] == 500 for r in rows), path
                values = {r['suite']: (r['successes'], r['valid_trials']) for r in rows}
                successes, trials = sum(r['successes'] for r in rows), sum(r['valid_trials'] for r in rows)
                assert trials == 2000 and math.isclose(successes/trials, data['total_average_success_rate']), path
            else:
                rows = data['category_results']
                assert {r['category'] for r in rows} == set(CATEGORIES), path
                assert all(r['trials'] == r['tasks_expected'] == r['tasks_complete'] for r in rows), path
                values = {r['category']: (r['successes'], r['trials']) for r in rows}
                successes, trials = sum(r['successes'] for r in rows), sum(r['trials'] for r in rows)
                assert trials == 10030 and successes == data['overall']['successes'], path
                assert math.isclose(successes/trials, data['overall']['success_rate']), path
            for r in rows:
                trial_count = r['valid_trials'] if benchmark == 'libero' else r['trials']
                assert 0 <= r['successes'] <= trial_count
                assert math.isclose(r['successes']/trial_count, r['success_rate']), path
            results[experiment, benchmark] = dict(values=values, total=(successes, trials), path=path)
    return results


def cell(counts):
    return f'{counts[0]/counts[1]:.2%}（{counts[0]}/{counts[1]}）'


def detailed_table(result, labels, heading):
    lines = [f'| {heading} | 成功数 | 有效 trial | 成功率 |', '|---|---:|---:|---:|']
    for label in [*labels, '总计']:
        successes, trials = result['total'] if label == '总计' else result['values'][label]
        lines.append(f'| {label} | {successes} | {trials} | **{successes/trials:.2%}** |')
    return '\n'.join(lines)


def comparison(results):
    lines = ['<!-- ENCODER_DETAIL_COMPARISON_BEGIN -->', '## LIBERO 各套件成功率对比', '',
             '单元格为成功率（成功数 / 有效 trial）；仅填写完整评测结果。', '']
    for benchmark, labels, title in [('libero', SUITES, '套件'), ('libero-plus', CATEGORIES, '扰动类别')]:
        if benchmark == 'libero-plus':
            lines += ['', '## LIBERO-Plus 各类别成功率对比', '',
                      '统一使用 corrected text；总成功率按全部 trial 加权计算，不对七个类别取简单平均。', '']
        names = []
        for experiment in SOURCES:
            result = results[experiment, benchmark]
            names.append(f'[{experiment}]({result["path"].relative_to(ROOT)})' if result else experiment)
        lines += ['| ' + title + ' | ' + ' | '.join(names) + ' |', '|---|---:|---:|---:|---:|']
        for label in [*labels, '总计']:
            cells = []
            for experiment in SOURCES:
                result = results[experiment, benchmark]
                cells.append(cell(result['total'] if label == '总计' else result['values'][label])
                             if result else '待完整评测')
            lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '<!-- ENCODER_DETAIL_COMPARISON_END -->']
    return '\n'.join(lines)


def experiment_block(experiment, results):
    prefix, version, patch = ('E03_PATCH', 'Wan2.1', 4) if experiment == 'E03' else ('E04', 'Wan2.2', 2)
    title = 'E03-Patch' if experiment == 'E03' else 'E04'
    run = ('outputs/m29_rynn9b_wan21_vae_patch4_cache/train/s42/2026-10-25_13-38-35'
           if experiment == 'E03' else 'outputs/m29_rynn9b_wan22_vae/train/s42/2026-10-26_10-03-52')
    lines = [f'<!-- {prefix}_AUTO_RESULT_BEGIN -->', f'## {title}：M29 + RynnBrain 9B + {version} VAE encoder', '',
             f'原始 VAE latent Cache + 空间 {patch}×{patch} Patch，882 visual tokens；从头训练完成 10,690 steps。', '',
             f'最终 checkpoint：`{run}/checkpoint_step_10690.safetensors`。', '']
    for benchmark, labels, heading in [('libero', SUITES, '套件'), ('libero-plus', CATEGORIES, '扰动类别')]:
        lines += ['### ' + ('LIBERO' if benchmark == 'libero' else 'LIBERO-Plus'), '']
        result = results[experiment, benchmark]
        if result is None:
            lines += ['使用 corrected text；评测尚未完整结束，类别成功率待完整评测后自动填写。', '']
            continue
        lines += [f'完整评测，非 smoke；总成功率：**{cell(result["total"])}**。错误/无效记录：0。', '']
        if benchmark == 'libero-plus':
            lines += ['使用 corrected text；总成功率按总成功数 / 总有效 trial 计算。', '']
        lines += [detailed_table(result, labels, heading), '',
                  f'原始结果：[summary.json]({result["path"].relative_to(ROOT)})。', '']
    lines += [f'<!-- {prefix}_AUTO_RESULT_END -->']
    return '\n'.join(lines)


def update():
    results = load_results()
    target = ROOT / 'result.md'
    with (QUEUE / 'result_details.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = target.read_text()
        start, end = '<!-- ENCODER_DETAIL_COMPARISON_BEGIN -->', '<!-- ENCODER_DETAIL_COMPARISON_END -->'
        block = comparison(results)
        if start in text:
            before, rest = text.split(start, 1); _, after = rest.split(end, 1)
            text = before + block + after
        else:
            position = text.index('\n## E02：')
            text = text[:position] + '\n\n' + block + '\n' + text[position:]
        for experiment, marker in [('E03', 'E03_PATCH'), ('E04', 'E04')]:
            start, end = f'<!-- {marker}_AUTO_RESULT_BEGIN -->', f'<!-- {marker}_AUTO_RESULT_END -->'
            block = experiment_block(experiment, results)
            if start in text:
                before, rest = text.split(start, 1); _, after = rest.split(end, 1)
                text = before + block + after
            else:
                assert experiment == 'E04'
                text, count = re.subn(r'(?ms)^## E04：.*?(?=^## |\Z)', lambda _: block+'\n', text, count=1)
                assert count == 1
        temporary = target.with_suffix('.md.details.tmp')
        temporary.write_text(text);temporary.replace(target)
    print('Updated complete LIBERO suites / LIBERO-Plus categories from verified summary counts.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow-e04', action='store_true')
    args = parser.parse_args()
    if args.follow_e04:
        with (QUEUE / 'result_details_watcher.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while not (QUEUE / 'DONE').exists():
                if (QUEUE / 'STOP').exists():
                    raise SystemExit('Evaluation stopped; leaving completed results intact')
                time.sleep(15)
            update()
            (QUEUE / 'RESULT_DETAILS_DONE').write_text(json.dumps({'updated_at': time.time()})+'\n')
    else:
        update()
