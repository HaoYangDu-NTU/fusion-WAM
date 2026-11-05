import torch
import torch.nn as nn

from vjepa_policy.experimental.action_only_policy import build_action_only_policy
from vjepa_policy.models.vjepa_policy import build_vjepa_policy
from vjepa_policy.models.wm_predictor import build_world_model_predictor


class CountingEncoder(nn.Module):
    embed_dim = 16

    def __init__(self):
        super().__init__()
        self.full_calls = 0
        self.masked_calls = 0

    def forward(self, clips, mask=None):
        if mask is None:
            self.full_calls += 1
            return clips
        self.masked_calls += 1
        return torch.stack([clips[index, row] for index, row in enumerate(mask)], dim=0)


def make_predictor():
    return build_world_model_predictor(
        img_size=(4, 4),
        patch_size=2,
        num_frames=4,
        tubelet_size=2,
        embed_dim=16,
        predictor_embed_dim=12,
        out_embed_dim=16,
        depth=2,
        num_heads=3,
        num_mask_tokens=2,
        lang_dim=8,
        proprio_dim=5,
        interpolate_rope=False,
    )


def build_policy(builder, encoder, predictor, **kwargs):
    return builder(
        encoder,
        predictor,
        action_dim=2,
        action_chunk_size=3,
        context_len=4,
        proprio_dim=5,
        action_hidden_size=12,
        action_num_layers=2,
        condition_num_heads=3,
        action_num_inference_steps=2,
        **kwargs,
    )


def make_batch():
    return (
        torch.randn(2, 8, 16),
        torch.randn(2, 4, 8),
        torch.ones(2, 4, dtype=torch.bool),
        [torch.arange(4).unsqueeze(0).expand(2, -1)],
        [torch.arange(4, 8).unsqueeze(0).expand(2, -1)],
        torch.randn(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.randn(2, 5),
    )


def test_action_only_matches_zero_weight_world_loss_and_gradients():
    torch.manual_seed(31)
    regular = build_policy(
        build_vjepa_policy,
        CountingEncoder(),
        make_predictor(),
        world_loss_weight=0.0,
    )
    torch.manual_seed(31)
    action_only = build_policy(
        build_action_only_policy,
        CountingEncoder(),
        make_predictor(),
    )
    regular.load_state_dict(action_only.state_dict(), strict=True)
    batch = make_batch()

    torch.manual_seed(47)
    regular_loss, regular_metrics = regular(batch)
    regular_loss.backward()
    torch.manual_seed(47)
    action_loss, action_metrics = action_only(batch)
    action_loss.backward()

    torch.testing.assert_close(action_loss, regular_loss)
    torch.testing.assert_close(
        action_metrics["loss_action"], regular_metrics["loss_action"]
    )
    assert action_metrics["loss_world"].item() == 0.0
    assert action_metrics["cos_sim"].item() == 0.0
    regular_parameters = dict(regular.named_parameters())
    for name, parameter in action_only.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad,
                regular_parameters[name].grad,
                rtol=1e-5,
                atol=1e-7,
            )


def test_action_only_skips_full_target_encoder_and_prediction_head():
    encoder = CountingEncoder()
    policy = build_policy(build_action_only_policy, encoder, make_predictor())
    prediction_head_calls = []
    handle = policy.predictor.predictor_proj.register_forward_hook(
        lambda *_args: prediction_head_calls.append(True)
    )
    try:
        policy(make_batch())
    finally:
        handle.remove()

    assert encoder.full_calls == 0
    assert encoder.masked_calls == 1
    assert prediction_head_calls == []


def test_action_only_state_dict_loads_into_serving_topology():
    action_only = build_policy(
        build_action_only_policy,
        CountingEncoder(),
        make_predictor(),
    )
    serving_topology = build_policy(
        build_vjepa_policy,
        CountingEncoder(),
        make_predictor(),
    )

    assert (
        type(action_only.predictor).__name__
        == type(serving_topology.predictor).__name__
    )
    serving_topology.predictor.load_state_dict(
        action_only.predictor.state_dict(), strict=True
    )
    serving_topology.action_expert.load_state_dict(
        action_only.action_expert.state_dict(), strict=True
    )
