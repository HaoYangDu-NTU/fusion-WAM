"""Frozen spatial encoder queue: predecessor LIBERO, cache, smoke, train, LIBERO."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from openwam.experiments import resolve_experiment, source_fingerprint
from scripts.e01_ablation_queue import Queue, SUITES, read, wait_for_completed_queue
from benchmarks.vla_harness.run_resumable import atomic_json


def write_report(queue, result=None, experiment="E07"):
    encoder, size, tokens = ("SigLIP-L/16", 256, 1280) if experiment == "E08" else ("DINOv3-L/16", 224, 980)
    lines = [f"## {experiment}：M29 + RynnBrain 9B + {encoder}", "",
             f"冻结 {encoder}，{size}×{size} 独立双视角；2 clean＋8 future，{tokens} visual tokens。",
             "最后层 1024D patch features，沿宽度拼接；clean 直接投影，future bottleneck256→M29 1536D。",
             "x/x、shift42/5、logit_normal、λ_video=λ_action=1；seed42、global batch256、10690 steps。",
             "原始冻结特征独立 Cache；4 workers/rank、每 worker 2 线程。标准 LIBERO 40×50=2000 trials。", "",
             "| 套件 | 成功数 / Trials | 成功率 |", "|---|---:|---:|"]
    for suite in SUITES:
        rate = result["suites"][suite] if result else None
        lines.append(f"| {suite} | {round(rate*500) if rate is not None else '待完成'} / 500 | "
                     + (f"{100*rate:.2f}%" if rate is not None else "待完成") + " |")
    lines.append("| Overall | " + (f"{result['successes']} / 2000 | **{100*result['overall']:.2f}%** |"
                                         if result else "待完成 / 2000 | 待完成 |"))
    if result:
        lines += ["", f"原始结果：[summary.json]({result['summary']})。"]
    text = "\n".join(lines) + "\n"
    (queue/"RESULTS.md").write_text(text)
    target = ROOT/"result.md"
    original = target.read_text()
    begin, end = f"<!-- {experiment}_AUTO_RESULT_BEGIN -->", f"<!-- {experiment}_AUTO_RESULT_END -->"
    block = begin + "\n\n" + text + "\n" + end
    original = (re.sub(re.escape(begin)+r".*?"+re.escape(end), lambda _: block, original, flags=re.S)
                if begin in original else original.rstrip()+"\n\n"+block+"\n")
    original = re.sub(r"^\| E01-LV2 \|.*$",
        "| E01-LV2 | 2 | 1 | 已取消 | 已取消 | 已取消 | 已取消 | 用户停止调参，未训练 |",
        original, flags=re.M)
    if result:
        row = f"| {experiment} | {encoder}；冻结、独立双视角、2 clean＋8 future，{tokens} tokens | **{100*result['overall']:.2f}%**（{result['successes']}/2000） | 未安排 | 标准 LIBERO 完成 |"
        original = re.sub(r"^\| "+re.escape(experiment)+r" \|.*$", lambda _: row, original, flags=re.M)
    target.write_text(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT/"reports/E07_training_20261102")
    parser.add_argument("--after-queue", type=Path, default=ROOT/"reports/E01_LV05_cache_w2_20261101")
    parser.add_argument("--experiment", choices=("E07", "E08"), default="E07")
    args = parser.parse_args()
    queue = args.queue.resolve()
    queue.mkdir(parents=True, exist_ok=True)
    with (queue/"controller.lock").open("a") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = Queue(queue)
        wait_for_completed_queue(state, args.after_queue)
        write_report(queue, None, args.experiment)
        profile = args.experiment+"-Cache"
        cfg, _, _ = resolve_experiment(profile)
        OmegaConf.save(cfg, queue/"resolved_config.yaml", resolve=True)
        if (queue/"DONE").exists():
            result = state.train_evaluate(profile, args.experiment)
            write_report(queue, result, args.experiment)
            return
        state.cleanup_orphaned_stages()
        state.wait_idle()
        with (ROOT/"outputs/train_eval.gpu.lock").open("a") as gpu_lock:
            while True:
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    state.status("waiting_for_experiment_lock")
                    state.check_stop()
                    time.sleep(30)
            state.resource_fd = gpu_lock.fileno()
            fingerprint = source_fingerprint()
            if (queue/"source.json").exists() and read(queue/"source.json")["sha256"] != fingerprint:
                raise RuntimeError("Source changed since encoder queue started; inspect before resuming")
            atomic_json(queue/"source.json", dict(sha256=fingerprint))
            state.run_stage(queue/"cache_build", [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=4", "scripts/precompute_image_features.py", "--experiment", args.experiment, "--build",
                "--batch-size", "8", "--loader-workers", "2", "--write-inflight", "2"])
            state.run_stage(queue/"cache_validation", [sys.executable, "scripts/validate_image_features.py", "--experiment", args.experiment])
            validation = read(Path(cfg.dataloader.frame_feature_cache_dir)/"validation.json")
            if not validation.get("passed"):
                raise RuntimeError("Frozen feature cache equivalence failed")
            result = state.train_evaluate(profile, args.experiment)
            write_report(queue, result, args.experiment)
            atomic_json(queue/"results.json", {args.experiment: result})
            state.status("complete", experiment=profile, benchmarks=["libero"])
            atomic_json(queue/"DONE", dict(finished_at=time.time()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        queue = Path(sys.argv[sys.argv.index("--queue")+1]) if "--queue" in sys.argv else ROOT/"reports/E07_training_20261102"
        queue.mkdir(parents=True, exist_ok=True)
        atomic_json(queue/"status.json", dict(state="failed", error=repr(exc), updated_at=time.time()))
        raise
