import torch
import pytest

from vjepa_policy.models.wm_predictor import WorldModelPredictor


def make_predictor(**overrides):
    config = {
        "img_size": (32, 48),
        "patch_size": 16,
        "num_frames": 4,
        "tubelet_size": 2,
        "embed_dim": 24,
        "predictor_embed_dim": 48,
        "depth": 3,
        "num_heads": 4,
        "num_mask_tokens": 2,
        "lang_dim": 20,
        "proprio_dim": 6,
    }
    config.update(overrides)
    return WorldModelPredictor(**config)


def make_inputs(batch_size=2):
    context_ids = torch.tensor([[0, 1, 2, 3, 4, 5]]).expand(batch_size, -1)
    future_ids = torch.tensor([[6, 7, 8, 9, 10, 11]]).expand(batch_size, -1)
    return {
        "context": torch.randn(batch_size, 6, 24),
        "masks_x": context_ids,
        "masks_y": future_ids,
        "language": torch.randn(batch_size, 4, 20),
        "language_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool).expand(
            batch_size, -1
        ),
        "proprio": torch.randn(batch_size, 6),
    }


def test_output_and_layer_aligned_kv_shapes():
    predictor = make_predictor().eval()
    prediction, predictor_kv = predictor(**make_inputs())
    assert prediction.shape == (2, 6, 24)
    assert len(predictor_kv) == 3
    for key, value in predictor_kv:
        assert key.shape == (2, 4, 6, 12)
        assert value.shape == (2, 4, 6, 12)
        assert key.is_contiguous()
        assert value.is_contiguous()


def test_masked_language_padding_does_not_affect_output():
    torch.manual_seed(3)
    predictor = make_predictor().eval()
    inputs = make_inputs(batch_size=1)
    with torch.no_grad():
        baseline = predictor(**inputs)
        inputs["language"][:, 2:] = torch.randn_like(inputs["language"][:, 2:]) * 1000
        changed = predictor(**inputs)
    torch.testing.assert_close(baseline[0], changed[0])
    for baseline_kv, changed_kv in zip(baseline[1], changed[1]):
        torch.testing.assert_close(baseline_kv[0], changed_kv[0])
        torch.testing.assert_close(baseline_kv[1], changed_kv[1])


def test_predictor_returns_every_layer_without_hidden_adapters():
    predictor = make_predictor(depth=4)
    _, predictor_kv = predictor(**make_inputs())
    assert len(predictor_kv) == 4
    assert not any("adapter" in name for name, _ in predictor.named_parameters())


def test_activation_checkpointing_backward_with_direct_kv():
    torch.manual_seed(5)
    predictor = make_predictor(use_activation_checkpointing=True).train()
    inputs = make_inputs(batch_size=1)
    inputs["context"].requires_grad_(True)
    prediction, predictor_kv = predictor(**inputs)
    loss = prediction.square().mean() + sum(
        key.square().mean() + value.square().mean()
        for key, value in predictor_kv
    )
    loss.backward()

    assert inputs["context"].grad is not None
    assert predictor.language_projection.weight.grad is not None
    assert predictor.proprio_projection.weight.grad is not None
    first_block = predictor.predictor_blocks[0]
    assert first_block.self_attn.qkv.weight.grad is not None
    assert first_block.cross_attn.q.weight.grad is not None
    assert first_block.ffn.fc1.weight.grad is not None


def test_partial_activation_checkpointing_counts_blocks(monkeypatch):
    calls = 0
    original_checkpoint = torch.utils.checkpoint.checkpoint

    def counted_checkpoint(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_checkpoint(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counted_checkpoint)
    predictor = make_predictor(activation_checkpointing_blocks=2).train()
    prediction, _ = predictor(**make_inputs(batch_size=1))
    prediction.square().mean().backward()

    assert calls == 2
    assert predictor.activation_checkpointing_blocks == 2


def test_partial_activation_checkpointing_preserves_outputs_and_gradients():
    torch.manual_seed(17)
    baseline = make_predictor().train()
    checkpointed = make_predictor(activation_checkpointing_blocks=2).train()
    checkpointed.load_state_dict(baseline.state_dict())
    inputs = make_inputs(batch_size=1)

    def run(model):
        model.zero_grad(set_to_none=True)
        prediction, predictor_kv = model(**inputs)
        loss = prediction.square().mean() + sum(
            key.square().mean() + value.square().mean()
            for key, value in predictor_kv
        )
        loss.backward()
        return prediction, predictor_kv, {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    baseline_prediction, baseline_kv, baseline_grads = run(baseline)
    checkpointed_prediction, checkpointed_kv, checkpointed_grads = run(checkpointed)

    torch.testing.assert_close(baseline_prediction, checkpointed_prediction, rtol=0, atol=0)
    for baseline_pair, checkpointed_pair in zip(baseline_kv, checkpointed_kv, strict=True):
        for baseline_tensor, checkpointed_tensor in zip(
            baseline_pair, checkpointed_pair, strict=True
        ):
            torch.testing.assert_close(baseline_tensor, checkpointed_tensor, rtol=0, atol=0)
    assert baseline_grads.keys() == checkpointed_grads.keys()
    for name in baseline_grads:
        torch.testing.assert_close(
            baseline_grads[name], checkpointed_grads[name], rtol=0, atol=0
        )


@pytest.mark.parametrize("blocks", [-1, 4])
def test_partial_activation_checkpointing_validates_range(blocks):
    with pytest.raises(ValueError, match="activation_checkpointing_blocks"):
        make_predictor(activation_checkpointing_blocks=blocks)


def test_predictor_is_randomly_initialized_without_film_or_timestep_path():
    torch.manual_seed(11)
    first = make_predictor()
    torch.manual_seed(12)
    second = make_predictor()

    assert all(parameter.requires_grad for parameter in first.parameters())
    assert not hasattr(first, "film_heads")
    assert not hasattr(first, "time_embedding")
    assert not torch.equal(
        first.predictor_blocks[0].self_attn.qkv.weight,
        second.predictor_blocks[0].self_attn.qkv.weight,
    )
    assert torch.count_nonzero(first.language_projection.weight) > 0
    assert torch.count_nonzero(first.proprio_projection.weight) > 0


def test_multiview_selection_ids_use_t_view_h_w_with_local_rope_coordinates():
    predictor = make_predictor(img_size=(32, 32), num_views=2)
    selection_ids = torch.arange(16).view(1, -1)

    torch.testing.assert_close(
        predictor._selection_to_view_ids(selection_ids),
        torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1] * 2]),
    )
    torch.testing.assert_close(
        predictor._selection_to_rope_ids(selection_ids),
        torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]]),
    )


def test_multiview_embeddings_are_added_after_projection_to_context_and_future():
    predictor = make_predictor(img_size=(32, 32), num_views=2, depth=1).eval()
    with torch.no_grad():
        predictor.predictor_embed.weight.zero_()
        predictor.predictor_embed.bias.zero_()
        predictor.mask_tokens[1].zero_()
        predictor.view_embedding.weight[0].fill_(1.0)
        predictor.view_embedding.weight[1].fill_(2.0)

    captured = {}

    def capture_block_input(_module, args):
        captured["tokens"] = args[0].detach().clone()

    def capture_rope_ids(_module, _args, kwargs):
        captured["rope_ids"] = kwargs["mask"].detach().clone()

    block_handle = predictor.predictor_blocks[0].register_forward_pre_hook(
        capture_block_input
    )
    attention_handle = predictor.predictor_blocks[
        0
    ].self_attn.register_forward_pre_hook(capture_rope_ids, with_kwargs=True)
    inputs = make_inputs(batch_size=1)
    inputs["context"] = torch.randn(1, 8, 24)
    inputs["masks_x"] = torch.arange(8).view(1, -1)
    inputs["masks_y"] = torch.arange(8, 16).view(1, -1)
    try:
        prediction, predictor_kv = predictor(**inputs)
    finally:
        block_handle.remove()
        attention_handle.remove()

    expected_view_values = torch.tensor([[1.0] * 4 + [2.0] * 4 + [1.0] * 4 + [2.0] * 4])
    torch.testing.assert_close(captured["tokens"][..., 0], expected_view_values)
    torch.testing.assert_close(
        captured["rope_ids"],
        torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]]),
    )
    assert captured["tokens"].shape == (1, 16, 48)
    assert prediction.shape == (1, 8, 24)
    assert predictor_kv[0][0].shape == (1, 4, 8, 12)
    assert predictor.view_embedding.weight.requires_grad
    prediction.square().mean().backward()
    assert predictor.view_embedding.weight.grad is not None
    assert torch.all(predictor.view_embedding.weight.grad.abs().sum(dim=1) > 0)


def test_single_view_keeps_legacy_state_dict_and_rope_ids():
    predictor = make_predictor()
    selection_ids = torch.arange(predictor.num_patches).view(1, -1)

    assert predictor.view_embedding is None
    assert not any(key.startswith("view_embedding.") for key in predictor.state_dict())
    torch.testing.assert_close(
        predictor._selection_to_rope_ids(selection_ids), selection_ids
    )
