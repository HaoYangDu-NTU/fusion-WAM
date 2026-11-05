import pytest
import torch
import torch.nn as nn

from vjepa_policy.models.wm_predictor import (
    build_world_model_predictor,
)
from vjepa_policy.models.vjepa_policy import (
    _local_view_masks,
    build_vjepa_policy,
    encode_proprio,
    encode_video_views,
    encoded_proprio_dim,
)
from vjepa_policy.models.action_expert import (
    ActionConditionEncoder,
    ActionExpert,
    AdaRMSNorm,
    PredictorKVActionAttention,
)


class ToyEncoder(nn.Module):
    embed_dim = 16

    def forward(self, clips, mask=None):
        if mask is None:
            return clips
        return torch.stack(
            [clips[index, row] for index, row in enumerate(mask)], dim=0
        )


class ToyVideoEncoder(nn.Module):
    patch_size = 2
    tubelet_size = 2

    def forward(self, clips, mask=None):
        batch_size = clips.shape[0]
        local_tokens = torch.arange(8, device=clips.device).view(1, 8, 1)
        batch_ids = torch.arange(batch_size, device=clips.device).view(-1, 1, 1)
        latents = (batch_ids * 100 + local_tokens).expand(-1, -1, 3).float()
        if mask is None:
            return latents
        return torch.stack(
            [latents[index, row] for index, row in enumerate(mask)], dim=0
        )


class TrainingFlagEncoder(ToyVideoEncoder):
    _vjepa_policy_training_false = True

    def __init__(self):
        super().__init__()
        self.training_arguments = []

    def forward(self, clips, mask=None, training=None):
        self.training_arguments.append(training)
        return super().forward(clips, mask)


def make_predictor(depth=2, proprio_dim=5):
    return build_world_model_predictor(
        img_size=(4, 4),
        patch_size=2,
        num_frames=4,
        tubelet_size=2,
        embed_dim=16,
        predictor_embed_dim=12,
        out_embed_dim=16,
        depth=depth,
        num_heads=3,
        num_mask_tokens=2,
        lang_dim=8,
        proprio_dim=proprio_dim,
        interpolate_rope=False,
    )


def make_expert(num_layers=2):
    return ActionExpert(
        action_dim=2,
        action_chunk_size=3,
        num_layers=num_layers,
        predictor_num_heads=3,
        predictor_head_dim=4,
        state_dim=5,
        hidden_size=12,
        lang_dim=8,
        condition_num_heads=3,
        num_inference_steps=2,
    )


def test_independent_encoder_output_is_temporal_then_view_major():
    clips = torch.zeros(1, 2, 3, 4, 4, 4)
    encoder = ToyVideoEncoder()

    target = encode_video_views(encoder, clips)
    context_mask = torch.arange(8).unsqueeze(0)
    context = encode_video_views(encoder, clips, context_mask)

    assert target.shape == (1, 16, 3)
    assert context.shape == (1, 8, 3)
    expected_target = torch.tensor(
        [0, 1, 2, 3, 100, 101, 102, 103, 4, 5, 6, 7, 104, 105, 106, 107]
    )
    torch.testing.assert_close(target[0, :, 0], expected_target.float())
    torch.testing.assert_close(context[0, :, 0], expected_target[:8].float())


def test_vjepa2_1_encoder_is_called_with_training_false():
    encoder = TrainingFlagEncoder()
    clips = torch.zeros(1, 3, 4, 4, 4)

    output = encode_video_views(encoder, clips)

    assert output.shape == (1, 8, 3)
    assert encoder.training_arguments == [False]


def test_local_view_masks_preserve_each_samples_token_order():
    mask = torch.tensor(
        [
            [0, 1, 4, 5, 8, 9, 12, 13],
            [1, 0, 5, 4, 9, 8, 13, 12],
        ]
    )

    local = _local_view_masks(mask, num_views=2, patches_per_view=4)

    torch.testing.assert_close(
        local,
        torch.tensor(
            [
                [0, 1, 4, 5],
                [0, 1, 4, 5],
                [1, 0, 5, 4],
                [1, 0, 5, 4],
            ]
        ),
    )


def test_local_view_masks_reject_uneven_per_sample_view_counts():
    mask = torch.tensor([[0, 1, 2, 4], [0, 4, 5, 6]])

    with pytest.raises(ValueError, match="each sample and view"):
        _local_view_masks(mask, num_views=2, patches_per_view=4)


def test_action_condition_tokens_exactly_match_predictor_semantics():
    torch.manual_seed(3)
    predictor = make_predictor()
    conditioner = ActionConditionEncoder(8, 5, 12)
    conditioner.language_norm.load_state_dict(predictor.language_norm.state_dict())
    conditioner.language_projection.load_state_dict(
        predictor.language_projection.state_dict()
    )
    conditioner.state_norm.load_state_dict(predictor.proprio_norm.state_dict())
    conditioner.state_projection.load_state_dict(
        predictor.proprio_projection.state_dict()
    )
    language = torch.randn(2, 4, 8)
    language_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    state = torch.randn(2, 5)

    expected = predictor._condition_tokens(language, language_mask, state)
    actual = conditioner(language, language_mask, state)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert not hasattr(conditioner, "language_position")
    assert not hasattr(conditioner, "token_type")


def test_action_expert_uses_dit_blocks_and_direct_predictor_kv():
    expert = make_expert()

    assert not hasattr(expert, "film")
    assert expert.num_layers == 2
    assert expert.num_heads == 3
    assert expert.head_dim == 4
    assert expert.read_full_predictor_kv is False
    for block in expert.blocks:
        assert isinstance(block.pre_attn_norm, AdaRMSNorm)
        assert isinstance(block.attn, PredictorKVActionAttention)
        assert isinstance(block.pre_ffn_norm, AdaRMSNorm)
        assert block.attn.q_proj.in_features == 12
        assert block.attn.q_proj.out_features == 12
        assert torch.count_nonzero(
            block.condition_attention.output_projection.weight
        ) == 0
        assert torch.count_nonzero(
            block.condition_attention.output_projection.bias
        ) == 0


def test_flow_schedule_is_seeded_and_bounded():
    expert = make_expert()
    torch.manual_seed(19)
    first = expert.sample_time(16, torch.device("cpu"), torch.float32)
    torch.manual_seed(19)
    second = expert.sample_time(16, torch.device("cpu"), torch.float32)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.all((first >= 0) & (first <= 1))
    assert expert.num_inference_steps == 2
    assert expert.num_timestep_buckets == 1000


def test_sincos_proprio_encoding_is_shared_by_both_experts():
    raw_state = torch.tensor([[0.0, torch.pi / 2, torch.pi]])
    encoded = encode_proprio(raw_state, "sincos")

    assert encoded_proprio_dim(3, "sincos") == 6
    torch.testing.assert_close(
        encoded,
        torch.cat((torch.sin(raw_state), torch.cos(raw_state)), dim=-1),
    )

    policy = build_vjepa_policy(
        ToyEncoder(),
        make_predictor(proprio_dim=6),
        action_dim=2,
        action_chunk_size=3,
        context_len=4,
        proprio_dim=3,
        proprio_encoding="sincos",
        action_hidden_size=12,
        action_num_layers=2,
        condition_num_heads=3,
        action_num_inference_steps=2,
    )
    assert policy.raw_proprio_dim == 3
    assert policy.encoded_proprio_dim == 6
    assert policy.predictor.proprio_projection.in_features == 6
    assert policy.action_expert.conditioner.state_projection.in_features == 6


def test_fixed_width_state_contract_rejects_sincos_at_build_time():
    with pytest.raises(ValueError, match="fixed-width state packing"):
        build_vjepa_policy(
            ToyEncoder(),
            make_predictor(proprio_dim=96),
            action_dim=2,
            action_chunk_size=3,
            context_len=4,
            proprio_dim=3,
            proprio_encoding="sincos",
            max_state_dim=48,
            action_hidden_size=12,
            action_num_layers=2,
            condition_num_heads=3,
            action_num_inference_steps=2,
        )


def test_action_expert_reads_only_layer_aligned_context_kv():
    expert = make_expert()
    seen_lengths = []
    handles = []
    for block in expert.blocks:
        handles.append(
            block.attn.register_forward_pre_hook(
                lambda _module, args: seen_lengths.append(args[1].shape[2])
            )
        )
    predictor_kv = tuple(
        (
            torch.randn(2, 3, 7, 4),
            torch.randn(2, 3, 7, 4),
        )
        for _ in range(2)
    )
    try:
        expert(
            predictor_kv,
            torch.randn(2, 3, 2),
            torch.randn(2, 4, 8),
            torch.ones(2, 4, dtype=torch.bool),
            torch.randn(2, 5),
            context_len=4,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert seen_lengths == [4, 4]


def test_policy_joint_backward_uses_explicit_action_condition_branch():
    torch.manual_seed(23)
    predictor = make_predictor()
    policy = build_vjepa_policy(
        ToyEncoder(),
        predictor,
        action_dim=2,
        action_chunk_size=3,
        context_len=4,
        proprio_dim=5,
        action_hidden_size=12,
        action_num_layers=2,
        condition_num_heads=3,
        action_num_inference_steps=2,
    )
    for block in policy.action_expert.blocks:
        nn.init.eye_(block.condition_attention.output_projection.weight)

    state = torch.randn(2, 5, requires_grad=True)
    batch = (
        torch.randn(2, 8, 16),
        torch.randn(2, 4, 8),
        torch.ones(2, 4, dtype=torch.bool),
        [torch.arange(4).unsqueeze(0).expand(2, -1)],
        [torch.arange(4, 8).unsqueeze(0).expand(2, -1)],
        torch.randn(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
        state,
    )

    loss, metrics = policy(batch)
    loss.backward()

    assert set(metrics) == {"loss", "loss_world", "loss_action", "cos_sim"}
    assert isinstance(policy.proprio_encoder, nn.Identity)
    assert not any(policy.proprio_encoder.parameters())
    assert state.grad is not None and state.grad.abs().sum() > 0
    conditioner = policy.action_expert.conditioner
    assert conditioner.state_projection.weight.grad is not None
    assert conditioner.state_projection.weight.grad.abs().sum() > 0
    assert policy.action_expert.blocks[0].attn.q_proj.weight.grad is not None
    assert predictor.predictor_blocks[0].self_attn.qkv.weight.grad is not None
