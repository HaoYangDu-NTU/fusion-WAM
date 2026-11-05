import torch

from vjepa_policy.models.wm_predictor import (
    WorldModelPredictor,
)
from vjepa_policy.models.backbone.modules import (
    RoPEAttention,
    rotate_queries_or_keys,
)
from vjepa_policy.models.vision_transformer import VisionTransformer


def _manual_rotation(x, pos, corrected):
    dim = x.shape[-1]
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega = 1.0 / 10000 ** (omega / (dim / 2.0))
    frequency = torch.einsum("..., f -> ... f", pos, omega)
    if corrected:
        sine = frequency.sin().repeat_interleave(2, dim=-1)
        cosine = frequency.cos().repeat_interleave(2, dim=-1)
    else:
        sine = frequency.sin().squeeze(-1).repeat(1, 1, 1, 2)
        cosine = frequency.cos().squeeze(-1).repeat(1, 1, 1, 2)
    pairs = x.unflatten(-1, (-1, 2))
    first, second = pairs.unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return x * cosine + rotated * sine


def _make_predictor(**overrides):
    config = {
        "img_size": (32, 48),
        "patch_size": 16,
        "num_frames": 4,
        "tubelet_size": 2,
        "embed_dim": 24,
        "predictor_embed_dim": 48,
        "depth": 2,
        "num_heads": 4,
        "num_mask_tokens": 2,
        "lang_dim": 20,
        "proprio_dim": 6,
    }
    config.update(overrides)
    return WorldModelPredictor(**config)


def test_corrected_rotation_duplicates_frequency_within_each_channel_pair():
    queries = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    positions = torch.tensor([1.0])

    corrected = rotate_queries_or_keys(
        queries, positions, corrected_frequency_pairing=True
    )
    expected = _manual_rotation(queries, positions, corrected=True)
    legacy = rotate_queries_or_keys(queries, positions)

    torch.testing.assert_close(corrected, expected)
    assert not torch.allclose(corrected, legacy)


def test_legacy_rotation_is_the_default_and_is_numerically_unchanged():
    queries = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    positions = torch.tensor([1.0])

    default = rotate_queries_or_keys(queries, positions)
    explicit_legacy = rotate_queries_or_keys(
        queries, positions, corrected_frequency_pairing=False
    )
    expected = _manual_rotation(queries, positions, corrected=False)

    torch.testing.assert_close(default, expected, rtol=0, atol=0)
    torch.testing.assert_close(explicit_legacy, expected, rtol=0, atol=0)


def test_corrected_predictor_rope_does_not_change_checkpoint_keys():
    torch.manual_seed(7)
    legacy = _make_predictor()
    corrected = _make_predictor(corrected_rope_frequency_pairing=True)

    corrected.load_state_dict(legacy.state_dict(), strict=True)
    assert all(
        not block.self_attn.corrected_frequency_pairing
        for block in legacy.predictor_blocks
    )
    assert all(
        block.self_attn.corrected_frequency_pairing
        for block in corrected.predictor_blocks
    )
    assert legacy.state_dict().keys() == corrected.state_dict().keys()


def test_encoder_and_shared_attention_default_to_legacy_pairing():
    attention = RoPEAttention(dim=96, num_heads=3)
    encoder = VisionTransformer(
        img_size=32,
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=96,
        depth=1,
        num_heads=3,
        use_rope=True,
    )

    assert attention.corrected_frequency_pairing is False
    assert encoder.blocks[0].attn.corrected_frequency_pairing is False


def test_corrected_attention_changes_post_rope_keys_but_not_values():
    torch.manual_seed(7)
    legacy = RoPEAttention(dim=96, num_heads=3, grid_size=2)
    corrected = RoPEAttention(
        dim=96,
        num_heads=3,
        grid_size=2,
        corrected_frequency_pairing=True,
    )
    corrected.load_state_dict(legacy.state_dict(), strict=True)
    tokens = torch.randn(1, 4, 96)
    positions = torch.arange(4).unsqueeze(0)

    legacy_output, legacy_key, legacy_value = legacy(
        tokens,
        mask=positions,
        T=1,
        H_patches=2,
        W_patches=2,
        return_kv=True,
    )
    corrected_output, corrected_key, corrected_value = corrected(
        tokens,
        mask=positions,
        T=1,
        H_patches=2,
        W_patches=2,
        return_kv=True,
    )

    torch.testing.assert_close(corrected_value, legacy_value, rtol=0, atol=0)
    assert not torch.allclose(corrected_key, legacy_key)
    assert not torch.allclose(corrected_output, legacy_output)
