"""Supervise the authorized final E05 evaluation without touching training."""
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from safetensors import safe_open
from benchmarks.vla_harness.run_resumable import atomic_json, resources_idle, process_identity
from benchmarks.vla_harness.run_train_eval import alive
from benchmarks.vla_harness.checkpoint_contract import validate_vggt_omega_checkpoint
from benchmarks.vla_harness.prepare_libero_plus import prepare_checkpoint
import yaml

MAIN = Path("/data/fcy/code/openwam-exp")
QUEUE = MAIN / "reports/E05_training/automatic_evaluation"
RUN = ROOT / "outputs/m29_rynn9b_vggt_omega_spatial/train/s42/2026-10-26_23-52-12"
CHECKPOINT = "checkpoint_step_10690.safetensors"
PLUS_CACHE = ROOT / "cache/text/rynnbrain1p1-9b/libero-plus/avg_all_layers/corrected_text"


def status(state, **details):
    atomic_json(QUEUE / "status.json", dict(state=state, updated_at=time.time(), **details))


def render_results(original):
    import importlib.util
    spec = importlib.util.spec_from_file_location("result_details", MAIN / "reports/E04_wan22/update_result_details.py")
    details = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(details)
    details.SOURCES["E05"] = QUEUE / "evaluation"
    results = details.load_results()
    standard, plus = results["E05", "libero"], results["E05", "libero-plus"]
    if standard is None or plus is None:
        raise RuntimeError("E05 full evaluations are incomplete")
    lines = ["<!-- E05_AUTO_RESULT_BEGIN -->", "## E05：M29 + RynnBrain 9B + VGGT-Ω", "",
             "冻结 VGGT-Ω 512 权重，224×224/视角；逐时间位置双视角联合编码，沿宽度拼接空间特征。",
             "2 clean＋8 future，980 visual tokens；从头训练完成 10,690 steps，global batch 256，seed 42。", "",
             f"最终 checkpoint：`{RUN / CHECKPOINT}`。", ""]
    for benchmark, labels, heading in (("libero", details.SUITES, "套件"),
                                        ("libero-plus", details.CATEGORIES, "扰动类别")):
        result = results["E05", benchmark]
        lines += ["### " + benchmark.upper(), "",
                  f"完整评测，非 smoke；总成功率：**{details.cell(result['total'])}**。错误/无效记录：0。", ""]
        if benchmark == "libero-plus":
            lines += ["使用 corrected text；按全部 trial 加权计算总成功率。", ""]
        lines += [details.detailed_table(result, labels, heading), "",
                  f"原始结果：[summary.json]({result['path'].relative_to(MAIN)})。", ""]
    lines.append("<!-- E05_AUTO_RESULT_END -->")
    block = "\n".join(lines)
    if "<!-- E05_AUTO_RESULT_BEGIN -->" in original:
        before, rest = original.split("<!-- E05_AUTO_RESULT_BEGIN -->", 1)
        _, after = rest.split("<!-- E05_AUTO_RESULT_END -->", 1)
        updated = before + block + after
    else:
        updated, count = re.subn(r"(?ms)^## E05：.*?(?=^## |\Z)", lambda _: block + "\n", original, count=1)
        if count != 1:
            raise RuntimeError("Missing E05 result section")
    row = ("| E05 | VGGT-Ω 512 Encoder；冻结、双视角联合编码、空间 Patch；2 clean＋8 future，980 tokens | "
           f"**{details.cell(standard['total'])}** | **{details.cell(plus['total'])}** | 两项完整评测完成 |")
    updated, count = re.subn(r"(?m)^\| E05 \|.*$", lambda _: row, updated, count=1)
    assert count == 1
    comparison = details.comparison(results).replace('|---|---:|---:|---:|---:|', '|---|---:|---:|---:|---:|---:|')
    start, end = '<!-- ENCODER_DETAIL_COMPARISON_BEGIN -->', '<!-- ENCODER_DETAIL_COMPARISON_END -->'
    before, rest = updated.split(start, 1); _, after = rest.split(end, 1)
    return before + comparison + after


def record_results():
    target = MAIN / "result.md"
    with (QUEUE / "result.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        updated = render_results(target.read_text())
        temporary = target.with_suffix(".md.e05.tmp")
        temporary.write_text(updated)
        temporary.replace(target)


def main():
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / "waiter.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (QUEUE / "DONE").exists():
            return
        identity_path = QUEUE / "training_process.json"
        if not identity_path.exists():
            launch = json.loads((QUEUE.parent / "formal_training_launch.json").read_text())
            atomic_json(identity_path, {"process": process_identity(launch["pid"]), "run": str(RUN)})
        process = json.loads(identity_path.read_text())["process"]
        while not (QUEUE / "STOP").exists():
            if process and alive(process):
                status("waiting_for_training", run=str(RUN))
                time.sleep(15)
                continue
            result = json.loads((RUN / "training_result.json").read_text())
            if result["status"] != "completed" or result["global_step"] != 10690:
                raise RuntimeError(f"Training did not complete: {result}")
            marker = json.loads((RUN / "accel_state_step_10690/trainer_state.json").read_text())
            if marker["global_step"] != 10690:
                raise RuntimeError("Final resume state is incomplete")
            with safe_open(RUN / CHECKPOINT, framework="np") as handle:
                if not handle.keys():
                    raise RuntimeError("Empty final checkpoint")
            cfg = yaml.safe_load((RUN / "config.yaml").read_text())
            validate_vggt_omega_checkpoint(RUN, CHECKPOINT, cfg)
            if not resources_idle("0,1,2,3", range(8990, 8998)):
                status("waiting_for_resources")
                time.sleep(15)
                continue
            prepare_checkpoint(RUN, CHECKPOINT,
                               RUN / "eval_libero_plus" / CHECKPOINT.removesuffix(".safetensors") / "corrected_text",
                               PLUS_CACHE, corrected=True)
            command = ["bash", "scripts/eval_libero_then_plus.sh", str(RUN), CHECKPOINT,
                       str(QUEUE / "evaluation"), "--gpus", "0,1,2,3", "--workers", "40",
                       "--max-batch-size", "16", "--render-gpus", "0,1,2,3", "--base-port", "8990",
                       "--plus-workers", "128", "--plus-max-batch-size", "32",
                       "--plus-render-gpus", "0,1,2,3", "--plus-base-port", "8994",
                       "--correct-instruction-suffix", "true", "--retry-delay", "30",
                       "--wait-for-idle", "--stop-file", str(QUEUE / "STOP")]
            atomic_json(QUEUE / "evaluation_command.json", command)
            status("evaluating", checkpoint=CHECKPOINT)
            env = {**os.environ, "OPENWAM_PYTHON": sys.executable, "TEXT_EMBEDDING_DIR": str(PLUS_CACHE),
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            env.pop("PLUS_DEPLOY_DIR", None)
            with (QUEUE / "launcher.log").open("a") as output:
                code = subprocess.call(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            if code or not (QUEUE / "evaluation/DONE").exists():
                raise RuntimeError(f"Evaluation supervisor exited {code}; inspect launcher.log")
            record_results()
            atomic_json(QUEUE / "DONE", dict(finished_at=time.time(), checkpoint=CHECKPOINT))
            status("complete", result=str(MAIN / "result.md"))
            return
        status("stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=repr(error))
        raise
