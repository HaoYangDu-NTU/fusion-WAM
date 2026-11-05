"""Authorized E01 prediction -> shift search, with standard LIBERO only.

--prepare audits/reuses E01 and writes the initial report without touching GPUs.
--run waits for idle GPUs and executes the ordered, resumable experiment queue.
No new baseline training and no LIBERO-Plus stage are scheduled.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from omegaconf import OmegaConf
from safetensors import safe_open
from openwam.experiments import resolve_experiment, runtime_env, source_fingerprint
from openwam.train.utils.checkpointing import find_latest_accel_state
from benchmarks.vla_harness.run_resumable import (
    atomic_json, resources_idle, process_identity, completed_tasks, OWNER_KEY, stop_owned_processes,
)
from benchmarks.vla_harness.run_train_eval import prepare_deploy

BASE_RUN = ROOT / "outputs/m29_rynn9b_dinov3/train/s42/2026-10-24_02-30-16"
BASE_EVAL = ROOT / "outputs/m29_rynn9b_dinov3/eval/s42/2026-10-24_02-30-16/libero"
CHECKPOINT = "checkpoint_step_10690.safetensors"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def read(path):
    return json.loads(Path(path).read_text())


def alive(record):
    actual = process_identity(record["pid"])
    return actual is not None and actual["state"] != "Z" and all(
        actual[k] == record[k] for k in ("starttime", "boot_id"))


def summary_result(path):
    data = read(path)
    if data.get("smoke") or not data.get("complete") or not data.get("selected_run_complete"):
        raise ValueError(f"Incomplete/smoke evaluation: {path}")
    suites = {s["suite"]: s for s in data["suites"]}
    if set(suites) != set(SUITES) or any(
        s["valid_trials"] != 500 or s["expected_trials"] != 500 or s["errors"] or not s["complete"]
        for s in suites.values()):
        raise ValueError("Expected four suites, 500 valid trials each and zero infrastructure errors")
    total = sum(s["successes"] for s in suites.values())
    if not math.isclose(data["total_average_success_rate"], total / 2000):
        raise ValueError("Summary success counts do not agree")
    return dict(summary=str(Path(path).resolve()), successes=total, trials=2000,
                overall=total/2000, suites={k: suites[k]["successes"]/500 for k in SUITES})


def audit_base():
    saved = yaml.safe_load((BASE_RUN / "config.yaml").read_text())
    cfg, _, _ = resolve_experiment("E01")
    current = OmegaConf.to_container(cfg, resolve=True)
    for section in ("dataloader", "training"):
        for key, value in saved[section].items():
            if key in current[section] and key not in ("output_path", "resume_ckpt_path", "finetune_ckpt_path"):
                if current[section][key] != value:
                    raise ValueError(f"Saved E01 differs from current baseline: {section}.{key}")
    checks = {
        "video_prediction": saved["model"]["architecture"]["video_prediction_type"] == "x",
        "action_prediction": saved["model"]["architecture"]["action_prediction_type"] == "x",
        "video_shift": saved["model"]["video_backbone"]["shift_video"] == 42,
        "action_shift": saved["model"]["action_backbone"]["shift_action"] == 5,
        "distribution": saved["training"]["noise_schedule_distribution"] == "logit_normal",
        "action_scale": saved["model"]["architecture"]["action_scale"] == 1,
        "seed": saved["project"]["seed"] == 42,
    }
    if not all(checks.values()):
        raise ValueError(f"Baseline reference audit failed: {checks}")
    # Compare all architectural knobs against the saved run, excluding generated component specs.
    for section in ("architecture", "action_backbone"):
        for key, value in current["model"][section].items():
            if key in saved["model"][section] and saved["model"][section][key] != value:
                raise ValueError(f"Saved baseline model differs: {section}.{key}")
    for section in ("dit_config", "encoder"):
        for key, value in current["model"]["video_backbone"][section].items():
            if saved["model"]["video_backbone"][section].get(key) != value:
                raise ValueError(f"Saved baseline video model differs: {section}.{key}")
    for key in ("name", "from_scratch", "shift_video", "text_encoder_type", "vlm_text_embedding_method", "text_embedding_dir"):
        if current["model"]["video_backbone"].get(key) != saved["model"]["video_backbone"].get(key):
            raise ValueError(f"Saved baseline visual/text configuration differs: {key}")
    if current["model"]["freeze"] != saved["model"]["freeze"]:
        raise ValueError("Saved baseline frozen modules differ")
    with safe_open(BASE_RUN / CHECKPOINT, framework="np") as tensors:
        tensor_count = len(tensors.keys())
        for key in tensors.keys():
            tensors.get_slice(key).get_shape()
    evaluation = read(BASE_EVAL / "run.json")
    plan = evaluation["plan"]
    if plan["benchmark"] != "libero" or plan["smoke"] or plan["num_trials"] != 50 or len(plan["tasks"]) != 40:
        raise ValueError("Saved baseline evaluation protocol differs")
    evaluated_weights = Path(evaluation["checkpoint_dir"]) / evaluation["checkpoint_name"]
    if not evaluated_weights.samefile(BASE_RUN / CHECKPOINT):
        raise ValueError("Baseline evaluation used different checkpoint weights")
    records = [episode for p in (BASE_EVAL / "completed").glob("*.json") for episode in read(p)["episodes"]]
    if len(completed_tasks(records, plan["tasks"], 50)) != 40:
        raise ValueError("Baseline does not have 40 unique complete task records")
    return dict(passed=True, reference_run=str(BASE_RUN), checkpoint=CHECKPOINT, tensor_count=tensor_count,
                config_sha256=hashlib.sha256((BASE_RUN / "config.yaml").read_bytes()).hexdigest(),
                base_result=summary_result(BASE_EVAL / "summary.json"), checks=checks,
                budget=dict(seed=42, per_gpu_batch=64, global_batch=256, steps=10690, epochs=10),
                cache_policy="Online E01 until production GPU feature-cache equivalence is validated",
                prediction_math="z=(1-sigma)*x+sigma*noise; v=noise-x; x->v=(z-x)/max(sigma,0.05)",
                shift_math="sigma_shift=s*sigma/(1+(s-1)*sigma); applies to train and inference schedules",
                metrics_note="Training future MSE/cosine use current noisy timesteps; not rollout forecasts")


def choose_prediction(results):
    """Conservative single-seed rule, fixed before results become available."""
    base = results["E01-Base"]
    winner = max(results, key=lambda name: results[name]["overall"])
    candidate = results[winner]
    se = math.sqrt((base["overall"]*(1-base["overall"]) + candidate["overall"]*(1-candidate["overall"])) / 2000)
    uncertain = candidate["overall"] - base["overall"] <= max(0.01, 1.96*se)
    conflict = any(candidate["suites"][s] < base["suites"][s] - 0.01 for s in ("libero_goal", "libero_10"))
    selected = "E01-Base" if uncertain or conflict else winner
    return dict(selected=selected, best_observed=winner, uncertain=uncertain, goal_long_conflict=conflict,
                minimum_gain=max(0.01, 1.96*se),
                rule="Keep x/x if <=1pp or within approximate 95% unpaired uncertainty, or Goal/Long drops >1pp; one seed only")


def write_report(queue, results, selection=None):
    modes = {"E01-Base": ("x", "x", 42), "E01-P1": ("x", "v", 42), "E01-P2": ("v", "x", 42),
             "E01-P3": ("v", "v", 42)}
    selected_modes = modes[selection["selected"]][:2] if selection else ("待第一阶段", "待第一阶段")
    modes.update({name: (*selected_modes, shift) for name,shift in (("E01-S12",12),("E01-S24",24),("E01-S42",42),("E01-S60",60))})
    lines = ["# E01 Prediction Target → Video Noise Shift", "",
             "标准 LIBERO；40 tasks × 50 trials。E01-Base 复用；S42 复用第一阶段选中结果。", "",
             "| Experiment | Video Pred | Action Pred | Video Shift | Spatial | Object | Goal | Long | LIBERO Overall |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for name,(v,a,shift) in modes.items():
        row=results.get(name)
        values=[f"{100*row['suites'][s]:.2f}%" for s in SUITES]+[f"{100*row['overall']:.2f}%"] if row else ["待完成"]*5
        lines.append(f"| {name} | {v} | {a} | {shift} | " + " | ".join(values) + " |")
    lines += ["", "预算：seed42、global batch256、10690 steps；冻结 DINO 和 RynnBrain；其余沿用保存的 E01。",
              "", "Prediction 的 x/v 均按既有 velocity-space flow loss 计算；不同参数化的 loss 数值不能直接当作策略优劣。",
              "训练中的 future MSE/cosine 只包含有效 future 位置，排除 clean prefix 与 padding；单独记录为诊断指标。",
              "", "P3 为新增 v/v 实验；其完整结果通过验收前，不能比较全部四种 prediction 组合。"]
    if selection:
        lines += ["", "第一阶段选择：`"+selection["selected"]+"`。"+selection["rule"],
                  "S42 为同一 checkpoint/结果的别名，不计为独立训练。"]
        base=results["E01-Base"]
        for name,stream in (("E01-P1","Action x→v，Video 固定 x"),("E01-P2","Video x→v，Action 固定 x")):
            delta=100*(results[name]["overall"]-base["overall"])
            lines += [f"- {stream}：Overall {delta:+.2f} 个百分点；仅代表该条件下的单 seed 观测。"]
        for name,row in results.items():
            goal=100*(row["suites"]["libero_goal"]-base["suites"]["libero_goal"])
            long=100*(row["suites"]["libero_10"]-base["suites"]["libero_10"])
            lines += [f"- {name}：Goal {goal:+.2f} pp，Long {long:+.2f} pp。"]
    diagnostic=[(name,row.get("training_metrics")) for name,row in results.items() if row.get("training_metrics")]
    if diagnostic:
        lines += ["", "最后 100 个唯一训练 step 的诊断均值：", "",
                  "| Experiment | Video Loss | Action Loss | Future MSE | Future Cosine |", "|---|---:|---:|---:|---:|"]
        for name,metrics in diagnostic:
            lines.append("| "+name+" | "+" | ".join(f"{metrics[k]:.6g}" for k in ("loss_video","loss_action","future_mse","future_cosine"))+" |")
        lines += ["", "E01-Base 未记录 future MSE/cosine，因此不能补写或直接用其缺失指标解释成功率。",
                  "各组噪声 shift/参数化不同，训练诊断只作为辅助证据；成功率优先。"]
    for name,row in results.items():
        lines += ["", f"- {name}: [{Path(row['summary']).name}]({row['summary']})"]
    if all(name in results for name in ("E01-S24","E01-S42","E01-S60")):
        completed_shifts=[name for name in ("E01-S12","E01-S24","E01-S42","E01-S60") if name in results]
        best=max(completed_shifts,key=lambda n:results[n]["overall"])
        gain=100*(results[best]["overall"]-results["E01-Base"]["overall"])
        lines += ["",f"Shift 组最高观测结果：{best}，相对 E01-Base {gain:+.2f} 个百分点。",
                  "单 seed、未做独立重复实验；Goal/Long 的套件差异与训练指标一起解读，不作因果保证。"]
    text="\n".join(lines)+"\n"
    (queue/"RESULTS.md").write_text(text)
    atomic_json(queue/"results.json",results)
    # Keep the project's result entry visible as requested, without touching historical tables.
    target=ROOT/"result.md"
    original=target.read_text()
    marker="<!-- E01_ABLATION_QUEUE_BEGIN -->"
    block=marker+"\n\n## E01 Prediction / Shift 消融\n\n"+"\n".join(lines[4:6+len(modes)])+f"\n\n完整记录：[RESULTS.md]({queue.relative_to(ROOT)}/RESULTS.md)。\n\n<!-- E01_ABLATION_QUEUE_END -->"
    original=re.sub(re.escape(marker)+r".*?<!-- E01_ABLATION_QUEUE_END -->",lambda _:block,original,flags=re.S) if marker in original else original.rstrip()+"\n\n"+block+"\n"
    if "E01-P3" in results:
        row=results["E01-P3"]
        completed=("<!-- E01_P3_QUEUE_BEGIN -->\n\nE01-P3（v/v）已完成训练与标准 LIBERO："
                   f"{row['successes']}/{row['trials']}，Overall **{100*row['overall']:.2f}%**。"
                   f"套件结果见上方消融表；[评测记录]({row['summary']})。\n\n<!-- E01_P3_QUEUE_END -->")
        original=re.sub(r"<!-- E01_P3_QUEUE_BEGIN -->.*?<!-- E01_P3_QUEUE_END -->",lambda _:completed,original,flags=re.S)
    target.write_text(original)
    if (queue/"video_loss_plan.json").exists() or any(name in results for name in ("E01-LV05", "E01-LV2")):
        write_video_loss_report(queue, results)


def write_video_loss_report(queue, results):
    """Keep loss-weight ablations separate from prediction/shift results."""
    lines = ["## E01 Video Loss 权重消融", "",
             "固定 x/x、video/action shift42/5、logit_normal、λ_action=1；仅改变 λ_video。",
             "新实验从头训练 10690 步，global batch256、seed42，复用 DINO 原始特征 Cache、worker 2 线程。",
             "每组先 smoke，再正式训练和标准 LIBERO 40×50=2000 trials；不运行 LIBERO-Plus。", "",
             "| Experiment | λ_video | λ_action | Spatial | Object | Goal | Long | LIBERO Overall |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, weight in (("E01-Base", 1), ("E01-LV05", 0.5), ("E01-LV2", 2)):
        row = results.get(name)
        values = ([f"{100*row['suites'][s]:.2f}%" for s in SUITES] + [f"{100*row['overall']:.2f}%"]
                  if row else ["待完成"]*5)
        lines.append(f"| {name} | {weight:g} | 1 | " + " | ".join(values) + " |")
    lines += ["", "λ_video=1 复用原 E01-Base，不新增基线训练。历史 baseline 在线编码；新实验采用已验证的冻结特征缓存。",
              "日志 Video Loss 已乘 λ_video；跨权重比较原始视频损失时须除以对应权重。Future MSE/cosine 不乘该权重。",
              "单训练种子结果，不代表统计显著性；总体成功率为主，同时检查 Goal/Long。"]
    for name in ("E01-LV05", "E01-LV2"):
        if name in results:
            row = results[name]
            lines += ["", f"- {name}: [{Path(row['summary']).name}]({row['summary']})"]
    text = "\n".join(lines) + "\n"
    (queue/"VIDEO_LOSS_RESULTS.md").write_text(text)
    with (queue/"RESULTS.md").open("a") as stream:
        stream.write("\n" + text)
    target = ROOT/"result.md"
    original = target.read_text()
    begin, end = "<!-- E01_VIDEO_LOSS_BEGIN -->", "<!-- E01_VIDEO_LOSS_END -->"
    block = begin + "\n\n" + text + f"\n完整记录：[VIDEO_LOSS_RESULTS.md]({queue.relative_to(ROOT)}/VIDEO_LOSS_RESULTS.md)。\n\n" + end
    original = (re.sub(re.escape(begin)+r".*?"+re.escape(end), lambda _: block, original, flags=re.S)
                if begin in original else original.rstrip()+"\n\n"+block+"\n")
    target.write_text(original)


def stage_worker(stage):
    stage=Path(stage)
    job=read(stage/"command.json")
    with (stage/"worker.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (stage/"exit.json").exists():
            return
        atomic_json(stage/"worker.json",process_identity(os.getpid()))
        code=1
        try:
            with (stage/"job.log").open("a") as log:
                child=subprocess.Popen(job["command"],cwd=ROOT,env={**os.environ,**runtime_env("0,1,2,3"),OWNER_KEY:job["owner"]},
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(stage/"child.json",process_identity(child.pid))
                code=child.wait()
        finally:
            stop_owned_processes(job["owner"])
            atomic_json(stage/"exit.json",dict(exit_code=code,finished_at=time.time()))


class Queue:
    def __init__(self,path,visual_cache=False):
        self.path=path
        self.visual_cache=visual_cache

    def status(self,state,**kwargs):
        atomic_json(self.path/"status.json",dict(state=state,updated_at=time.time(),**kwargs))
        print(state,kwargs,flush=True)

    def check_stop(self):
        if (self.path/"STOP").exists():
            for path in self.path.glob("*/**/command.json"):
                if not (path.parent/"exit.json").exists():
                    stop_owned_processes(read(path)["owner"])
            raise InterruptedError("Queue STOP requested")

    def wait_idle(self):
        while True:
            self.check_stop()
            if resources_idle("0,1,2,3",range(8940,8944)):
                return
            self.status("waiting_for_current_gpu_tasks")
            time.sleep(30)

    def cleanup_orphaned_stages(self):
        """Recover only this queue's recorded process owners after a controller failure."""
        for path in self.path.glob("*/**/command.json"):
            stage=path.parent
            worker=read(stage/"worker.json") if (stage/"worker.json").exists() else None
            if worker and alive(worker):
                continue
            job=read(path)
            stop_owned_processes(job["owner"])
            if "benchmarks.vla_harness.run_resumable" in job["command"]:
                command=job["command"]
                nested=Path(command[command.index("--output-dir")+1])/"job.json"
                if nested.exists():
                    stop_owned_processes(read(nested)["job_id"])

    def run_stage(self,stage,command):
        stage.mkdir(parents=True,exist_ok=True)
        if (stage/"exit.json").exists():
            if read(stage/"exit.json")["exit_code"] == 0:
                return stage/"job.log"
            raise RuntimeError(f"Failed stage: {stage}; retain failure and resume from a new attempt")
        if not (stage/"command.json").exists():
            self.wait_idle()
            atomic_json(stage/"command.json",dict(command=command,owner=uuid.uuid4().hex,created_at=time.time()))
        if read(stage/"command.json")["command"] != command:
            raise RuntimeError(f"Stage command changed: {stage}")
        worker=read(stage/"worker.json") if (stage/"worker.json").exists() else None
        if worker and not alive(worker):
            atomic_json(stage/"exit.json",dict(exit_code=-1,error="Worker disappeared; checkpoint recovery required"))
            raise RuntimeError(f"Worker disappeared: {stage}")
        spawn=read(stage/"spawn.json") if (stage/"spawn.json").exists() else None
        pending_worker=not worker and spawn and spawn.get("process") and alive(spawn["process"])
        if not worker and spawn and not pending_worker:
            raise RuntimeError(f"Reserved stage lost its worker: {stage}; inspect before retrying")
        if not worker and not pending_worker:
            with (stage/"supervisor.log").open("a") as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--stage-worker",str(stage)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(self.resource_fd,))
                atomic_json(stage/"spawn.json",dict(process=process_identity(child.pid),started_at=time.time()))
        self.status("running",stage=str(stage),command=command)
        while not (stage/"exit.json").exists():
            self.check_stop()
            worker=read(stage/"worker.json") if (stage/"worker.json").exists() else None
            if worker and not alive(worker):
                time.sleep(2)
                if not (stage/"exit.json").exists():
                    raise RuntimeError(f"Worker died: {stage}")
            time.sleep(10)
        if read(stage/"exit.json")["exit_code"]:
            raise RuntimeError(f"Stage failed: {stage}; see job.log")
        return stage/"job.log"

    def train_evaluate(self,name,label):
        if self.visual_cache:
            name += "-Cache"
        folder=self.path/label
        folder.mkdir(exist_ok=True)
        done=folder/"accepted.json"
        if done.exists():
            previous=read(done)
            return {**previous, **summary_result(previous["summary"])}
        prepared=ROOT/"reports/E01_P1_visual_cache_20261030"
        reuse_smoke=(name=="E01-P1-Cache" and (prepared/"DONE.json").exists())
        if reuse_smoke:
            for stage in ("smoke","smoke_inference"):
                if read(prepared/stage/"exit.json")["exit_code"] != 0:
                    raise RuntimeError("Prepared P1 cache smoke did not complete")
            smoke=prepared/"smoke/job.log"
        else:
            smoke=self.run_stage(folder/"smoke",[sys.executable,"scripts/experiment.py","train",name,
                                 "--gpus","0,1,2,3","--smoke-steps","4"])
        checkpoint=final_checkpoint(smoke)
        prepare_deploy(checkpoint.parent)
        # Strict real checkpoint loading and one actual standard-LIBERO rollout.
        inference=prepared/"smoke_inference/job.log" if reuse_smoke else self.run_stage(folder/"smoke_inference",[sys.executable,"-m","benchmarks.vla_harness.run_eval",
            "--benchmark","libero","--ckpt-dir",str(checkpoint.parent),"--ckpt-name",checkpoint.name,
            "--gpus","0","--workers","1","--max-batch-size","1","--render-gpus","3",
            "--suites","libero_spatial","--task-ids","0","--smoke","--base-port","8940",
            "--correct-instruction-suffix","false"])
        # Check all ranks completed smoke, finite losses and useful gradients.
        launch=re.search(r"Launch report: ([^\r\n]+)",smoke.read_text(errors="replace"))
        if not launch:
            raise RuntimeError("Missing smoke launch report")
        metrics=[]
        for rank in range(4):
            rows=[json.loads(line) for line in (Path(launch[1])/"timing"/f"rank{rank}.jsonl").read_text().splitlines()]
            if [r["step"] for r in rows] != [1,2,3,4] or not all(math.isfinite(v) for r in rows for v in r["metrics"].values()):
                raise RuntimeError("Invalid/nonfinite smoke metrics")
            if not any(r["metrics"].get("grad_norm",0)>0 for r in rows):
                raise RuntimeError("Smoke has no gradients")
            metrics.append(rows[-1])
        atomic_json(folder/"smoke_acceptance.json",dict(passed=True,checkpoint=str(checkpoint),ranks=metrics,
                    inference_log=str(inference),formal_result=False))
        cfg,_,_=resolve_experiment(name)
        runs=sorted(Path(cfg.training.output_path).glob("*/config.yaml"))
        if len(runs)>1:
            raise RuntimeError("Ambiguous formal runs; manual inspection required")
        run=runs[0].parent if runs else None
        result=read(run/"training_result.json") if run and (run/"training_result.json").exists() else {}
        if result.get("status")!="completed" or result.get("global_step")!=10690:
            command=[sys.executable,"scripts/experiment.py","train",name,"--gpus","0,1,2,3"]
            attempts=sorted(folder.glob("training_attempt_*"))
            active=attempts and not (attempts[-1]/"exit.json").exists()
            if run and not active:
                if not find_latest_accel_state(str(run)):
                    raise RuntimeError(f"Incomplete formal run has no full resume state: {run}")
                command += ["--resume",str(run)]
            if active:
                stage=attempts[-1]
                command=read(stage/"command.json")["command"]
            else:
                stage=folder/f"training_attempt_{len(attempts)+1:02d}"
            training=self.run_stage(stage,command)
            checkpoint=final_checkpoint(training)
            if checkpoint.name!=CHECKPOINT:
                raise RuntimeError("Formal budget not completed")
            run=checkpoint.parent
        else:
            checkpoint=run/CHECKPOINT
        prepare_deploy(run)
        evaluation=[sys.executable,"-m","benchmarks.vla_harness.run_resumable","--benchmarks","libero",
            "--ckpt-dir",str(run),"--ckpt-name",checkpoint.name,"--output-dir",str(folder/"evaluation"),
            "--gpus","0,1,2,3","--workers","32","--max-batch-size","8","--render-gpus","3",
            "--base-port","8940","--correct-instruction-suffix","false","--wait-for-idle",
            "--stop-file",str(self.path/"STOP")]
        attempts=sorted(folder.glob("evaluation_attempt_*"))
        if not (folder/"evaluation/libero/summary.json").exists() or not read(folder/"evaluation/libero/summary.json").get("complete"):
            stage=attempts[-1] if attempts and not (attempts[-1]/"exit.json").exists() else folder/f"evaluation_attempt_{len(attempts)+1:02d}"
            self.run_stage(stage,evaluation)
        result=summary_result(folder/"evaluation/libero/summary.json")
        metrics_path=run/"ablation_metrics.jsonl"
        if metrics_path.exists():
            unique={}
            for line in metrics_path.read_text().splitlines():
                record=json.loads(line)
                unique[record["step"]]=record
            tail=[unique[k] for k in sorted(unique)[-100:]]
            result["training_metrics"]={key:sum(r[key] for r in tail)/len(tail)
                for key in ("loss_video","loss_action","future_mse","future_cosine")}
        atomic_json(done,dict(**result,run=str(run),checkpoint=checkpoint.name))
        return result


def final_checkpoint(logfile):
    text=Path(logfile).read_text(errors="replace")
    if re.search(r"Traceback \(most recent call last\)|ChildFailedError|CUDA out of memory",text):
        raise RuntimeError(f"Training error: {logfile}")
    values=re.findall(r"\[checkpoint\] Saved final: ([^\r\n]+)",text)
    if len(set(values))!=1:
        raise RuntimeError(f"Missing/ambiguous final checkpoint: {logfile}")
    checkpoint=Path(values[-1].strip())
    if not checkpoint.is_absolute():
        checkpoint=ROOT/checkpoint
    with safe_open(checkpoint,framework="np") as tensors:
        for key in tensors.keys():
            tensors.get_slice(key).get_shape()
    return checkpoint


def wait_for_completed_queue(state, dependency):
    """Require predecessor completion and valid full LIBERO results before launch."""
    dependency=dependency.resolve()
    if dependency == state.path.resolve():
        raise ValueError("A queue cannot depend on itself")
    while True:
        state.check_stop()
        status=read(dependency/"status.json") if (dependency/"status.json").exists() else {}
        if (dependency/"STOP").exists() or status.get("state") in ("failed","stopped_by_user","cancelled"):
            raise RuntimeError(f"Predecessor stopped or failed: {dependency}")
        if (dependency/"DONE").exists() and status.get("state")=="complete":
            results=read(dependency/"results.json")
            if not results:
                raise RuntimeError("Predecessor has no accepted results")
            for row in results.values():
                summary_result(row["summary"])
            atomic_json(state.path/"dependency_acceptance.json",dict(
                queue=str(dependency),accepted_at=time.time(),results=list(results),benchmarks=["libero"]))
            return results
        state.status("waiting_for_predecessor",dependency=str(dependency),benchmarks=["libero"])
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue",type=Path,default=ROOT/"reports/E01_ablation_20261030")
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--visual-cache",action="store_true",help="Use validated DINO cache profiles for all new runs")
    parser.add_argument("--single-experiment",help="Run only this explicit experiment profile, then stop")
    parser.add_argument("--result-label",help="Result row name for --single-experiment")
    parser.add_argument("--after-queue",type=Path,help="Wait for this queue's completed, validated LIBERO results")
    parser.add_argument("--stage-worker",type=Path)
    args=parser.parse_args()
    if args.result_label and not args.single_experiment:
        parser.error("--result-label requires --single-experiment")
    if args.single_experiment and args.visual_cache:
        parser.error("--single-experiment takes an explicit cache profile; omit --visual-cache")
    if args.after_queue and not args.single_experiment:
        parser.error("--after-queue requires --single-experiment")
    if args.stage_worker:
        stage_worker(args.stage_worker)
        return
    queue=args.queue.resolve()
    queue.mkdir(parents=True,exist_ok=True)
    with (queue/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=Queue(queue,visual_cache=args.visual_cache)
        inherited=wait_for_completed_queue(state,args.after_queue) if args.after_queue and args.run else {}
        audit=audit_base()
        if args.visual_cache:
            audit["cache_policy"]="Validated raw frozen DINO frame cache; Projection and original policy modules remain online"
        if args.single_experiment:
            cfg,profile,_=resolve_experiment(args.single_experiment)
            audit["requested_experiment"]=args.single_experiment
            audit["worker_threads"]=int(getattr(cfg.training,"dataloader_worker_threads",1))
            if profile.get("data_overrides",{}).get("frame_feature_cache_dir"):
                audit["cache_policy"]="Validated raw frozen DINO frame cache; Projection and original policy modules remain online"
        atomic_json(queue/"baseline_audit.json",audit)
        results={**inherited, **(read(queue/"results.json") if (queue/"results.json").exists() else {})}
        results.setdefault("E01-Base",audit["base_result"])
        selection=read(queue/"selection.json") if (queue/"selection.json").exists() else None
        write_report(queue,results,selection)
        if not args.run:
            stages=[args.single_experiment] if args.single_experiment else ["P1","P2","selection","S24","S60"]
            state.status("prepared",base_reused=True,stages=stages,benchmarks=["libero"])
            return
        state.cleanup_orphaned_stages()
        state.wait_idle()
        resource=ROOT/"outputs/train_eval.gpu.lock"
        resource.parent.mkdir(exist_ok=True)
        with resource.open("a") as gpu_lock:
            while True:
                try:
                    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    state.status("waiting_for_experiment_lock")
                    state.check_stop()
                    time.sleep(30)
            state.resource_fd=gpu_lock.fileno()
            audit_base()  # Recheck the reference after a potentially long resource wait.
            fingerprint=source_fingerprint()
            if (queue/"source.json").exists() and read(queue/"source.json")["sha256"]!=fingerprint:
                raise RuntimeError("Code/config changed since queue execution started; inspect before resuming")
            atomic_json(queue/"source.json",dict(sha256=fingerprint))
            if args.single_experiment:
                label=args.result_label or args.single_experiment
                results[label]=state.train_evaluate(args.single_experiment,label)
                write_report(queue,results,selection)
                state.status("complete",experiment=args.single_experiment,benchmarks=["libero"])
                atomic_json(queue/"DONE",dict(finished_at=time.time()))
                return
            for name in ("E01-P1","E01-P2"):
                results[name]=state.train_evaluate(name,name)
                write_report(queue,results)
            selection=choose_prediction({k:results[k] for k in ("E01-Base","E01-P1","E01-P2")})
            atomic_json(queue/"selection.json",selection)
            modes={"E01-Base":"xx","E01-P1":"xv","E01-P2":"vx"}
            results["E01-S42"]={**results[selection["selected"]],"reused_from":selection["selected"]}
            write_report(queue,results,selection)
            for shift in (24,60):
                label=f"E01-S{shift}"
                results[label]=state.train_evaluate(f"{label}-{modes[selection['selected']]}",label)
                write_report(queue,results,selection)
            state.status("complete",base_reused=True,s42_reused=selection["selected"],benchmarks=["libero"])
            atomic_json(queue/"DONE",dict(finished_at=time.time()))


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        if "--stage-worker" not in sys.argv:
            parser_queue=Path(sys.argv[sys.argv.index("--queue")+1]) if "--queue" in sys.argv else ROOT/"reports/E01_ablation_20261030"
            parser_queue.mkdir(parents=True,exist_ok=True)
            atomic_json(parser_queue/"status.json",dict(state="failed",error=repr(exc),updated_at=time.time()))
        raise
