import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from openwam.experiments import resolve_experiment
from scripts.e01_ablation_queue import choose_prediction, summary_result
from tests.test_flow_prediction import TinyArchitecture


def result(overall, goal=None, long=None):
    return dict(overall=overall,suites=dict(libero_goal=goal or overall,libero_10=long or overall))


def test_prediction_selection_keeps_base_when_uncertain_or_conflicting():
    rows={"E01-Base":result(.9435),"E01-P1":result(.9500),"E01-P2":result(.9410)}
    assert choose_prediction(rows)["selected"]=="E01-Base"
    rows["E01-P1"]=result(.9700)
    assert choose_prediction(rows)["selected"]=="E01-P1"
    rows["E01-P1"]=result(.9700,goal=.9200)
    assert choose_prediction(rows)["selected"]=="E01-Base"


@pytest.mark.parametrize("name,v,a,shift",[("E01-P1","x","v",42),("E01-P2","v","x",42),
    ("E01-S24-vx","v","x",24),("E01-S60-xv","x","v",60)])
def test_profiles_change_only_requested_policy_and_output(name,v,a,shift):
    base=OmegaConf.to_container(resolve_experiment("E01")[0],resolve=True)
    cfg=OmegaConf.to_container(resolve_experiment(name)[0],resolve=True)
    assert cfg["model"]["architecture"]["video_prediction_type"]==v
    assert cfg["model"]["architecture"]["action_prediction_type"]==a
    assert cfg["model"]["video_backbone"]["shift_video"]==shift
    cfg["model"]["architecture"]["video_prediction_type"]="x"
    cfg["model"]["architecture"]["action_prediction_type"]="x"
    cfg["model"]["video_backbone"]["shift_video"]=42.0
    assert cfg["model"]==base["model"]
    for k in ("output_path","report_future_quality"):
        cfg["training"].pop(k,None);base["training"].pop(k,None)
    assert cfg["training"]==base["training"]
    assert cfg["dataloader"]==base["dataloader"]
    assert cfg["project"]["seed"]==42


@pytest.mark.parametrize("v,a",[("x","v"),("v","x"),("v","v")])
def test_future_diagnostics_do_not_change_training_loss_or_gradients(v,a):
    arch=TinyArchitecture({"video_prediction_type":v,"action_prediction_type":a})
    inputs=dict(actions=torch.ones(2,2,3),input_latents=torch.ones(2,1,3,2,2),
        clean_condition_latents=torch.ones(2,1,1,2,2),num_clean_condition_latents=1,
        video_is_pad=torch.tensor([[False,True],[False,False]]))
    outputs=[]
    for enabled in (False,True):
        arch._report_future_quality=enabled
        arch.zero_grad(set_to_none=True)
        torch.manual_seed(42)
        loss=arch.compute_loss(**copy.deepcopy(inputs))
        loss["loss"].backward()
        outputs.append((loss,arch.weight.grad.clone()))
    torch.testing.assert_close(outputs[0][0]["loss"],outputs[1][0]["loss"],rtol=0,atol=0)
    torch.testing.assert_close(outputs[0][1],outputs[1][1],rtol=0,atol=0)
    assert torch.isfinite(outputs[1][0]["future_mse"])
    assert torch.isfinite(outputs[1][0]["future_cosine"])


def test_acceptance_rejects_smoke_and_incomplete_trials(tmp_path):
    path=tmp_path/"summary.json"
    data=dict(smoke=False,complete=True,selected_run_complete=True,total_average_success_rate=.9,
        suites=[dict(suite=s,valid_trials=500,expected_trials=500,successes=450,errors=0,complete=True)
                for s in ("libero_spatial","libero_object","libero_goal","libero_10")])
    path.write_text(json.dumps(data))
    assert summary_result(path)["trials"]==2000
    data["smoke"]=True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):summary_result(path)
    data["smoke"]=False;data["suites"][0]["valid_trials"]=499
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):summary_result(path)


def test_resumable_controller_libero_only_never_calls_plus(tmp_path,monkeypatch):
    import sys
    from benchmarks.vla_harness import run_resumable as controller
    checkpoint=tmp_path/"checkpoint";checkpoint.mkdir()
    (checkpoint/"config.yaml").write_text("model: {}\n")
    (checkpoint/"weights.safetensors").write_bytes(b"test")
    called=[]
    monkeypatch.setattr(controller,"run_stage",lambda args,job,stage:called.append(stage) or True)
    monkeypatch.setattr(controller,"stop_owned_processes",lambda job:None)
    monkeypatch.setattr(sys,"argv",["run_resumable","--benchmarks","libero","--ckpt-dir",str(checkpoint),
        "--ckpt-name","weights.safetensors","--output-dir",str(tmp_path/"evaluation")])
    assert controller.main()==0
    assert called==["libero"]


def full_summary():
    return dict(smoke=False,complete=True,selected_run_complete=True,total_average_success_rate=.9,
        suites=[dict(suite=s,valid_trials=500,expected_trials=500,successes=450,errors=0,complete=True)
                for s in ("libero_spatial","libero_object","libero_goal","libero_10")])


def test_dependency_waits_for_recorded_full_evaluation(tmp_path,monkeypatch):
    import scripts.e01_ablation_queue as queue
    predecessor=tmp_path/"p2";predecessor.mkdir()
    successor=tmp_path/"p3";successor.mkdir()
    queue.atomic_json(predecessor/"status.json",dict(state="running"))
    sleeps=[]
    def complete_after_wait(seconds):
        sleeps.append(seconds)
        queue.atomic_json(predecessor/"summary.json",full_summary())
        queue.atomic_json(predecessor/"results.json",{"E01-P2":{"summary":str(predecessor/"summary.json")}})
        queue.atomic_json(predecessor/"status.json",dict(state="complete"))
        queue.atomic_json(predecessor/"DONE",{})
    monkeypatch.setattr(queue.time,"sleep",complete_after_wait)
    results=queue.wait_for_completed_queue(queue.Queue(successor),predecessor)
    assert sleeps==[30] and list(results)==["E01-P2"]
    assert (successor/"dependency_acceptance.json").exists()


@pytest.mark.parametrize("invalid",["failed","smoke","incomplete"])
def test_dependency_rejects_failed_or_invalid_predecessor(tmp_path,invalid):
    import scripts.e01_ablation_queue as queue
    predecessor=tmp_path/"p2";predecessor.mkdir()
    successor=tmp_path/"p3";successor.mkdir()
    data=full_summary()
    if invalid=="smoke": data["smoke"]=True
    if invalid=="incomplete": data["suites"][0]["valid_trials"]=499
    queue.atomic_json(predecessor/"summary.json",data)
    queue.atomic_json(predecessor/"results.json",{"E01-P2":{"summary":str(predecessor/"summary.json")}})
    queue.atomic_json(predecessor/"status.json",dict(state="failed" if invalid=="failed" else "complete"))
    queue.atomic_json(predecessor/"DONE",{})
    with pytest.raises((RuntimeError,ValueError)):
        queue.wait_for_completed_queue(queue.Queue(successor),predecessor)
    assert not (successor/"dependency_acceptance.json").exists()


def test_report_includes_p3_and_all_shift_rows(tmp_path,monkeypatch):
    import scripts.e01_ablation_queue as queue
    monkeypatch.setattr(queue,"ROOT",tmp_path)
    (tmp_path/"result.md").write_text("Existing results\n")
    destination=tmp_path/"queue";destination.mkdir()
    queue.write_report(destination,{})
    text=(tmp_path/"result.md").read_text()
    assert "| E01-P3 | v | v | 42 |" in text
    assert "| E01-S60 |" in text
    assert "| E01-S12 |" in text
    with (tmp_path/"result.md").open("a") as stream:
        stream.write("\n<!-- E01_P3_QUEUE_BEGIN -->pending<!-- E01_P3_QUEUE_END -->\n")
    summary=tmp_path/"summary.json";summary.write_text(json.dumps(full_summary()))
    queue.write_report(destination,{"E01-P3":summary_result(summary)})
    text=(tmp_path/"result.md").read_text()
    assert "pending" not in text and "1800/2000" in text


def test_shift_comparison_includes_completed_s12(tmp_path,monkeypatch):
    import scripts.e01_ablation_queue as queue
    monkeypatch.setattr(queue,"ROOT",tmp_path)
    (tmp_path/"result.md").write_text("Existing results\n")
    destination=tmp_path/"queue";destination.mkdir()
    summary=tmp_path/"summary.json";summary.write_text(json.dumps(full_summary()))
    row=summary_result(summary)
    results={name:dict(row) for name in ("E01-Base","E01-P1","E01-P2","E01-S12","E01-S24","E01-S42","E01-S60")}
    results["E01-S12"]["overall"]=.95
    queue.write_report(destination,results,{"selected":"E01-Base","rule":"fixed x/x"})
    assert "Shift 组最高观测结果：E01-S12" in (destination/"RESULTS.md").read_text()
    assert "| E01-S12 | x | x | 12 |" in (tmp_path/"result.md").read_text()
