"""After E08 standard LIBERO: E07 Plus, E08 Plus, record results, stop."""
import argparse
from collections import Counter
import fcntl
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e01_ablation_queue import Queue, read, summary_result, wait_for_completed_queue
from benchmarks.vla_harness.run_resumable import atomic_json, completed_tasks
from benchmarks.vla_harness.prepare_libero_plus_svae import prepare_checkpoint
from benchmarks.vla_harness.run_train_eval import prepare_deploy
from openwam.experiments import source_fingerprint

COUNTS = {"Objects Layout": 1525, "Camera Viewpoints": 1599, "Robot Initial States": 1550,
          "Language Instructions": 1537, "Light Conditions": 1142,
          "Background Textures": 1076, "Sensor Noise": 1601}
STANDARD_QUEUES = {"E07": ROOT/"reports/E07_training_20261102", "E08": ROOT/"reports/E08_training_20261102"}
TEXT_CACHE = ROOT/"cache/text/rynnbrain1p1-9b/libero-plus/avg_all_layers/corrected_text"


def plus_result(path):
    data = read(path)
    if (data.get("smoke") or not data.get("complete") or not data.get("selected_run_complete")
            or data.get("invalid_records") != 0 or data.get("expected_full_trials") != 10030):
        raise ValueError("Expected complete non-smoke Plus with 10030 valid trials")
    categories = {row["category"]: row for row in data["category_results"]}
    if len(data["category_results"]) != 7 or set(categories) != set(COUNTS):
        raise ValueError("Expected all seven Plus categories exactly once")
    for name, count in COUNTS.items():
        row = categories[name]
        if (row["trials"] != count or row["tasks_complete"] != count or row["tasks_expected"] != count
                or not 0 <= row["successes"] <= count
                or not math.isclose(row["success_rate"], row["successes"]/count)):
            raise ValueError(f"Invalid Plus category counts/rate: {name}")
    successes = sum(row["successes"] for row in categories.values())
    total = data["overall"]
    if (total["trials"] != 10030 or total["successes"] != successes
            or total["tasks_complete"] != 10030 or total["tasks_expected"] != 10030
            or not math.isclose(total["success_rate"], successes/10030)
            or not math.isclose(data["total_average_success_rate"], successes/10030)):
        raise ValueError("Plus Overall does not match category counts")
    return dict(summary=str(Path(path).resolve()), successes=successes, trials=10030,
                overall=successes/10030, categories=categories)


def validate_records(stage, result):
    catalog = read(stage/"task_catalog.json")
    expected = {(row["suite"], row["task_id"]): row["category"] for row in catalog}
    if len(catalog) != 10030 or len(expected) != 10030 or dict(Counter(expected.values())) != COUNTS:
        raise ValueError("Plus catalog is incomplete or duplicated")
    manifest = read(stage/"run.json")
    if manifest["plan"].get("correct_instruction_suffix") is not True:
        raise ValueError("Plus requires corrected instructions")
    seen, successes = set(), Counter()
    for path in (stage/"completed").glob("*.json"):
        row = read(path)
        key = row["suite"], row["task_id"]
        if key not in expected or key in seen or not completed_tasks(row["episodes"], [dict(suite=key[0], task_id=key[1])], 1):
            raise ValueError(f"Invalid or duplicate Plus completion: {path}")
        seen.add(key)
        successes[expected[key]] += int(row["episodes"][0]["metrics"]["success"])
    if seen != set(expected) or any(successes[name] != row["successes"] for name, row in result["categories"].items()):
        raise ValueError("Committed Plus episodes do not agree with summary")


def write_report(queue, results):
    lines = ["## E07 / E08：LIBERO-Plus", "",
             "完整七类10030条；corrected instructions，冻结原final step10690权重；10步sync、每chunk执行10步动作、关闭velocity cache。", "",
             "| 类别 | E07 DINOv3-L | E08 SigLIP-L |", "|---|---:|---:|"]
    for name, count in COUNTS.items():
        values = [f"{100*results[exp]['categories'][name]['success_rate']:.2f}%（{results[exp]['categories'][name]['successes']}/{count}）"
                  if exp in results else "待完成" for exp in STANDARD_QUEUES]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    values = [f"**{100*results[exp]['overall']:.2f}%**（{results[exp]['successes']}/10030）"
              if exp in results else "待完成" for exp in STANDARD_QUEUES]
    lines.append("| Overall | " + " | ".join(values) + " |")
    for exp, row in results.items():
        lines += ["", f"{exp} 原始结果：[summary.json]({row['summary']})。"]
    text = "\n".join(lines)+"\n"
    (queue/"RESULTS.md").write_text(text)
    atomic_json(queue/"results.json", results)
    target = ROOT/"result.md"
    original = target.read_text()
    begin, end = "<!-- E07_E08_PLUS_BEGIN -->", "<!-- E07_E08_PLUS_END -->"
    block = begin+"\n\n"+text+"\n"+end
    original = (re.sub(re.escape(begin)+r".*?"+re.escape(end), lambda _: block, original, flags=re.S)
                if begin in original else original.rstrip()+"\n\n"+block+"\n")
    for exp in STANDARD_QUEUES:
        def update(match):
            columns = match[0].split("|")
            if exp in results:
                row = results[exp]
                columns[4] = f" **{100*row['overall']:.2f}%**（{row['successes']}/10030） "
                columns[5] = " LIBERO / LIBERO-Plus均完成 "
            else:
                columns[4] = " 已安排，待完成 "
            return "|".join(columns)
        original = re.sub(r"^\| "+exp+r" \|.*$", update, original, flags=re.M)
    target.write_text(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT/"reports/E07_E08_plus_20261103")
    args = parser.parse_args()
    queue = args.queue.resolve()
    queue.mkdir(parents=True, exist_ok=True)
    with (queue/"controller.lock").open("a") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = Queue(queue)
        wait_for_completed_queue(state, STANDARD_QUEUES["E08"])
        results = read(queue/"results.json") if (queue/"results.json").exists() else {}
        write_report(queue, results)
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
                raise RuntimeError("Code changed since Plus queue started; inspect before resuming")
            atomic_json(queue/"source.json", dict(sha256=fingerprint))
            for exp, source_queue in STANDARD_QUEUES.items():
                source = read(source_queue/exp/"accepted.json")
                summary_result(source["summary"])
                deployment = prepare_checkpoint(source["run"], source["checkpoint"], queue/exp/"deployment", TEXT_CACHE, corrected=True)
                prepare_deploy(deployment)
                stage = queue/exp/"evaluation/libero-plus"
                accepted = queue/exp/"accepted.json"
                if accepted.exists():
                    result = plus_result(read(accepted)["summary"])
                else:
                    performance = (read(queue/"performance.json") if (queue/"performance.json").exists()
                                   else dict(workers=128, max_batch_size=32, dynamic_tasks=False))
                    command = [sys.executable, "-m", "benchmarks.vla_harness.run_resumable", "--benchmarks", "libero-plus",
                        "--ckpt-dir", str(deployment), "--ckpt-name", source["checkpoint"],
                        "--output-dir", str(queue/exp/"evaluation"), "--gpus", "0,1,2,3",
                        "--workers", str(performance["workers"]), "--max-batch-size", str(performance["max_batch_size"]), "--render-gpus", "0,1,2,3",
                        "--base-port", "8990", "--plus-base-port", "8994", "--correct-instruction-suffix", "true",
                        "--wait-for-idle", "--stop-file", str(queue/"STOP")]
                    if performance.get("dynamic_tasks"):
                        command.append("--dynamic-tasks")
                    attempts = sorted((queue/exp).glob("evaluation_attempt_*"))
                    attempt = attempts[-1] if attempts and not (attempts[-1]/"exit.json").exists() else queue/exp/f"evaluation_attempt_{len(attempts)+1:02d}"
                    state.run_stage(attempt, command)
                    result = plus_result(stage/"summary.json")
                validate_records(stage, result)
                atomic_json(accepted, dict(**result, source_run=source["run"], checkpoint=source["checkpoint"], corrected=True))
                results[exp] = result
                write_report(queue, results)
            state.status("complete", benchmarks=["libero-plus"], experiments=list(STANDARD_QUEUES))
            atomic_json(queue/"DONE", dict(finished_at=time.time()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        queue = Path(sys.argv[sys.argv.index("--queue")+1]) if "--queue" in sys.argv else ROOT/"reports/E07_E08_plus_20261103"
        queue.mkdir(parents=True, exist_ok=True)
        atomic_json(queue/"status.json", dict(state="failed", error=repr(exc), updated_at=time.time()))
        raise
