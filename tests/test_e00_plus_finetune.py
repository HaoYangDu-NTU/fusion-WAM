"""Guard the pilot's baseline identity, matched budgets and active sparsity."""
import copy
import json

from omegaconf import OmegaConf
import pytest

from openwam.model.architectures.utils.action_router import ActionVisualRouter
from scripts.e00_plus_finetune import ROOT, plan_config, training_command, validate_source, write_results


@pytest.fixture
def plan():
    return OmegaConf.load(ROOT / "configs/e00_plus_finetune.yaml")


@pytest.mark.parametrize("key,value", [
    ("model.video_backbone.text_encoder_type", "umt5-xxl"),
    ("model.video_backbone.encoder.seperate_encode_view", False),
    ("model.architecture.video_attention_mask_mode", "full"),
    ("dataloader.video_stride", 2),
])
def test_reject_semantically_different_baseline(plan, key, value):
    baseline, _, _ = plan_config(plan, "B0")
    source = copy.deepcopy(baseline)
    OmegaConf.update(source, key, value)
    with pytest.raises(ValueError, match="not the E00 baseline"):
        validate_source(source, baseline)


def test_baseline_batch_history_is_allowed_but_router_source_is_not(plan):
    baseline, _, _ = plan_config(plan, "B0")
    source = copy.deepcopy(baseline)
    source.training.batch_size = 32
    validate_source(source, baseline)
    source.model.architecture.action_router.enabled = True
    with pytest.raises(ValueError, match="already has a router"):
        validate_source(source, baseline)


def test_dense_sparse_match_and_regularizer_activates_within_budget(plan):
    dense, _, _ = plan_config(plan, "B1")
    sparse, _, _ = plan_config(plan, "B2")
    a = OmegaConf.to_container(dense.model.architecture.action_router)
    b = OmegaConf.to_container(sparse.model.architecture.action_router)
    assert a.pop("sparse_weight") == 0
    assert b.pop("sparse_weight") == .01
    assert a == b
    router = ActionVisualRouter(4, 4, **{k: v for k, v in b.items() if k != "enabled"})
    assert router.coefficient(0) == 0
    assert router.coefficient(plan.sparse_warmup) == 0
    assert 0 < router.coefficient(300) < .01
    assert router.coefficient(500) == router.coefficient(plan.steps - 1) == .01


@pytest.mark.parametrize("variant", ("B0", "B1", "B2"))
def test_commands_preserve_matched_training_and_isolate_smoke(plan, variant):
    formal = training_command(plan, variant)
    smoke = training_command(plan, variant, smoke=True)
    assert f"training.max_steps={plan.steps}" in formal
    assert f"training.batch_size={plan.batch_size_per_gpu}" in formal
    assert f"training.learning_rate={plan.learning_rate}" in formal
    assert "training.save_full_states_for_resume=true" in formal
    assert f"training.max_steps={plan.smoke_steps}" in smoke
    assert f"training.batch_size={plan.batch_size_per_gpu}" in smoke
    assert "training.save_full_states_for_resume=false" in smoke
    assert any("/initialization/" + variant in x for x in formal if "finetune_ckpt_path=" in x)
    assert not any("resume_ckpt_path=" in x and not x.endswith("=null") for x in formal)


def test_report_excludes_incomplete_results(tmp_path):
    for name, complete in (("initial_eval", False), ("eval_B0", True)):
        directory = tmp_path / name
        (directory / "evaluation").mkdir(parents=True)
        (directory / "DONE.json").write_text("{}")
        (directory / "evaluation/summary.json").write_text(json.dumps({
            "complete": complete, "selected_run_complete": complete,
            "suites": [{"successes": 1900, "valid_trials": 2000}],
            "total_average_success_rate": .95,
        }))
    write_results(tmp_path)
    report = (tmp_path / "RESULTS.md").read_text()
    assert "initial_eval" not in report
    assert "eval_B0 | 1900/2000 | 95.00%" in report
