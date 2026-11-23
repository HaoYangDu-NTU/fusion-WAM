"""Drive the E00 sequence in tmux and post its final report to the requesting Codex thread."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from e00_router_sequence import OUT, ROOT, PYTHON, read, write

CODEX = shutil.which('codex') or '/data/haoyang/.local/bin/codex'


def agent(args, log, timeout):
    with log.open('a') as handle:
        try:
            return subprocess.run([CODEX, 'exec', *args], cwd=ROOT, stdin=subprocess.DEVNULL,
                                  stdout=handle, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return 124


def main(output=OUT, controller="e00_router_sequence.py", title="E00 三阶段实验"):
    output.mkdir(parents=True, exist_ok=True)
    thread = sys.argv[1]
    with (output/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output/'supervisor.json', {'pid':os.getpid(),'return_thread':thread})
        failures = 0
        while not (output/'DONE.json').exists():
            with (output/'controller.log').open('a') as log:
                rc = subprocess.call([PYTHON,str(ROOT/'scripts'/controller)],cwd=ROOT,
                                     stdout=log,stderr=subprocess.STDOUT)
            if rc == 0 and (output/'DONE.json').is_file():
                break
            failures += 1
            if failures > 3:
                break
            prompt = (
                f'User-authorized maintenance of {title} in {ROOT}. '
                f'Read {output}/status.json, controller.log and latest attempt logs. Diagnose and directly fix '
                'recoverable runtime or automation errors. Preserve weights, raw outputs and all unrelated processes. '
                'Preserve locked training/evaluation design, comparisons and success checks. Do not start any jobs, '
                'additional agents or tmux sessions; the supervisor retries automatically after your turn. '
                'If an external dependency blocks progress, state the evidence. Do not message any other party.'
            )
            agent(['-s','workspace-write','--ephemeral',prompt],output/'recovery_agent.log',1800)
            time.sleep(60)
        if (output/'delivered.json').exists():
            return
        if (output/'DONE.json').exists():
            body = (output/'REPORT.md').read_text()
            prompt = (f'这是用户授权的后台实验脚本回报。{title}已完成。请现在在这个原会话里用中文汇报测量结果，'
                      '引用下方报告，解释门控是否值得保留，不启动新工作。报告正文如下。\n' + body)
        else:
            body = str(read(output/'status.json'))
            prompt = ('这是用户授权的实验后台通知。队列多次恢复后仍遇到阻塞。请在原会话如实汇报已完成部分及阻塞，'
                      '明确尚未完成的阶段。状态如下。\n'+body)
        prompt += f'\n本地报告目录 {output}'
        for delay in (0, 60, 300, 900):
            if delay:
                time.sleep(delay)
            rc = agent(['-s','read-only','resume','--skip-git-repo-check','-o',str(output/'delivered_message.txt'),
                        thread,prompt],output/'delivery.log',600)
            if rc == 0 and (output/'delivered_message.txt').is_file():
                write(output/'delivered.json',{'returncode':0,'thread':thread,'time':time.time()})
                return
        write(output/'delivery_failed.json',{'returncode':rc,'thread':thread})


if __name__ == '__main__':
    main()
