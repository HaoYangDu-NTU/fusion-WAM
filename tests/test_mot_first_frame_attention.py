"""Exact first-frame attention partition, including checkpointed joint blocks."""
from types import SimpleNamespace

import pytest
import torch

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver, _CleanConditionAttention
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan21


@pytest.mark.parametrize('first,length', [(1, 9), (4, 15), (8, 8)])
def test_partition_matches_masked_output_and_qkv_gradients(first, length):
    torch.manual_seed(91)
    driver = object.__new__(DualSystemMoTDriver)
    driver.num_heads = 2
    qkv = [torch.randn(2, length, 16, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    refs = [x.detach().clone().requires_grad_(True) for x in qkv]
    mask = torch.ones(length, length, dtype=torch.bool)
    mask[:first, first:] = False
    expected = driver._mixed_attention(*refs, mask)
    actual = driver._mixed_attention(*qkv, _CleanConditionAttention(first))
    weight = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, qkv, weight)
    reference_grads = torch.autograd.grad(expected, refs, weight)
    for x, y in zip([actual, *grads], [expected, *reference_grads]):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)


def _driver():
    dit = WanModel(dim=32, in_dim=4, out_dim=4, ffn_dim=64, num_heads=4,
                   num_layers=2, text_dim=16, freq_dim=8, eps=1e-6,
                   seperated_timestep=True, patch_size=(1, 2, 2), has_image_input=False)
    vb = Wan21(SimpleNamespace(dit=dit))
    ab = ActionDiT(action_dim=7, dim=32, ffn_dim=64, num_heads=4, num_layers=2,
                   video_dim=32, bridge_layers=(0, 1), variant='joint_self_attn', text_dim=16)
    return DualSystemMoTDriver(vb, ab, attention_mask_mode='mutual', video_attention_mask_mode='clean_condition_causal')


@pytest.mark.parametrize('num_clean', [1, 2])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_joint_blocks_and_parameter_gradients_match_dense_reference(checkpoint, monkeypatch, num_clean):
    # Exercise the production branch with a small CPU model; routing at the
    # real sequence sizes is checked independently below.
    monkeypatch.setattr('openwam.model.architectures.dual_system.mot_driver._CLEAN_CONDITION_SPLIT_MIN_TOKENS', 0)
    torch.manual_seed(32)
    driver = _driver()
    vb, ab = driver.vb, driver.ab
    seen = []
    original_mixed = driver._mixed_attention

    def record(q, k, v, mask):
        seen.append(isinstance(mask, _CleanConditionAttention))
        return original_mixed(q, k, v, mask)

    monkeypatch.setattr(driver, '_mixed_attention', record)

    def run():
        torch.manual_seed(12)
        vb.zero_grad(set_to_none=True)
        ab.zero_grad(set_to_none=True)
        latents = torch.randn(2, 4, 3, 4, 4, requires_grad=True)
        context = torch.randn(2, 4, 16)
        vstate = vb.prepare(latents=latents, timestep=torch.tensor([127., 719.]), context=context,
                            fuse_vae_embedding_in_latents=True, num_clean_condition_latents=num_clean)
        astate = ab.prepare_state(torch.randn(2, 3, 7), torch.tensor([127., 719.]), context=context)
        vs, acs = driver.run_joint_loop(vstate, astate, use_gradient_checkpointing=checkpoint)
        output = torch.cat([vs.hidden_states, acs.payload.x_action], dim=1)
        (output * torch.randn_like(output)).sum().backward()
        grads = {name: p.grad.detach().clone() for prefix, module in [('v', vb), ('a', ab)]
                 for key, p in module.named_parameters() if p.grad is not None for name in [prefix + key]}
        grads['latents'] = latents.grad.clone()
        return output.detach(), grads

    fast, fast_grads = run()
    assert seen and all(seen)
    seen.clear()
    original_mask = vb.build_video_to_video_mask
    # Custom mask implementations must keep the generic mask path.
    monkeypatch.setattr(vb, 'build_video_to_video_mask', lambda **kwargs: original_mask(**kwargs))
    dense, dense_grads = run()
    assert seen and not any(seen)
    torch.testing.assert_close(fast, dense, atol=3e-6, rtol=3e-6)
    assert fast_grads.keys() == dense_grads.keys()
    for name in fast_grads:
        torch.testing.assert_close(fast_grads[name], dense_grads[name], atol=3e-5, rtol=3e-5, msg=name)


@pytest.mark.parametrize('mode,video_mode', [
    ('action_sees_video', 'clean_condition_causal'), ('isolated', 'clean_condition_causal'),
    ('video_sees_action', 'clean_condition_causal'), ('mutual', 'per_frame_causal'), ('mutual', 'bidirectional'),
])
def test_other_masks_keep_dense_path(mode, video_mode, monkeypatch):
    driver = _driver()
    driver.attention_mask_mode = mode
    driver.vb.video_attention_mask_mode = video_mode
    context = torch.randn(1, 4, 16)
    vstate = driver.vb.prepare(latents=torch.randn(1, 4, 3, 4, 4), timestep=torch.ones(1), context=context,
                               fuse_vae_embedding_in_latents=True)
    astate = driver.ab.prepare_state(torch.randn(1, 3, 7), torch.ones(1), context=context)
    original = driver._mixed_attention
    def check(q, k, v, mask):
        assert isinstance(mask, torch.Tensor)
        return original(q, k, v, mask)
    monkeypatch.setattr(driver, '_mixed_attention', check)
    driver.run_joint_loop(vstate, astate)


@pytest.mark.parametrize('height,width,fast', [(7, 14, False), (12, 10, False), (24, 20, True)])
def test_partition_dispatch_at_training_sequence_lengths(height, width, fast, monkeypatch):
    driver = _driver()
    context = torch.randn(1, 4, 16)
    vstate = driver.vb.prepare(latents=torch.randn(1, 4, 5, height * 2, width * 2),
                               timestep=torch.ones(1), context=context, fuse_vae_embedding_in_latents=True)
    astate = driver.ab.prepare_state(torch.randn(1, 32, 7), torch.ones(1), context=context)
    seen = []
    def record(layer, vs, acs, *, attn_mask, **kwargs):
        assert isinstance(attn_mask, _CleanConditionAttention)
        seen.append(attn_mask.dense_mask is None)
        if attn_mask.dense_mask is not None:
            reference = driver._build_attention_mask(
                s_video=5 * height * width, s_action=32,
                video_tokens_per_frame=height * width, device=context.device,
            )
            torch.testing.assert_close(attn_mask.dense_mask, reference)
        return vs, acs
    monkeypatch.setattr(driver, 'step', record)
    driver.run_joint_loop(vstate, astate)
    assert seen == [fast] * driver.num_layers
