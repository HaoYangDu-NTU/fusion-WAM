import pytest
import torch

from openwam.model.architectures.dual_system import joint_self_attn as joint
from tests.test_mot_first_frame_attention import _driver


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('proprio', [False, True])
def test_fixed_text_preserves_joint_outputs_and_all_gradients(checkpoint, proprio, monkeypatch):
    torch.manual_seed(33)
    driver = _driver()
    model = joint.DualSystemSelfAttnArchitecture()
    model.video_backbone = driver.vb
    model.action_backbone = driver.ab
    model._mot_driver = driver
    model._init_proprio_context({'use_proprioception': proprio, 'state_dim': 3}, text_dim=16)
    shapes = []
    original_prepare = driver.vb.prepare
    def record(**kwargs):
        shapes.append(kwargs['context'].shape[1])
        return original_prepare(**kwargs)
    monkeypatch.setattr(driver.vb, 'prepare', record)

    def run():
        torch.manual_seed(89)
        model.zero_grad(set_to_none=True)
        context = torch.randn(2, 512, 16, requires_grad=True)
        # Include an entirely masked sample to pin safe all-masked behavior.
        mask = torch.zeros(2, 512, dtype=torch.bool)
        mask[0, :7] = True
        video, action = model(
            torch.randn(2, 3, 7), torch.tensor([211., 334.]),
            proprio=torch.randn(2, 3) if proprio else None,
            latents=torch.randn(2, 4, 3, 4, 4), timestep=torch.tensor([131., 687.]),
            context=context, context_mask=mask, fuse_vae_embedding_in_latents=True,
            use_gradient_checkpointing=checkpoint,
        )
        output = torch.cat([video.flatten(), action.flatten()])
        (output * torch.randn_like(output)).sum().backward()
        grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        grads['context'] = context.grad.clone()
        return output.detach(), grads

    actual, grads = run()
    assert shapes[-1] == 128 + int(proprio)
    monkeypatch.setattr(joint, '_fixed_wan_text_context', lambda context, mask, seq_lens=None: (context, mask))
    expected, ref_grads = run()
    assert shapes[-1] == 512 + int(proprio)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], atol=3e-5, rtol=3e-5, msg=name)


@pytest.mark.parametrize("length", [7, 128, 512])
@pytest.mark.parametrize("valid", [0, 6, 200])
def test_fixed_length_independent_of_valid_tokens(length, valid):
    context = torch.randn(2, length, 16)
    seq_lens = torch.tensor([min(valid, length), 0])
    fixed, mask = joint._fixed_wan_text_context(context, seq_lens=seq_lens)
    assert fixed.shape == (2, 128, 16)
    assert mask.shape == (2, 128)
    assert mask[0].sum() == min(valid, length, 128)
    assert not mask[1].any()
    torch.testing.assert_close(fixed[:, :min(length, 128)], context[:, :128])
    if length < 128:
        assert not fixed[:, length:].any()
        assert not mask[:, length:].any()


def test_missing_mask_marks_only_original_tokens_valid():
    context = torch.randn(2, 7, 16)
    _, mask = joint._fixed_wan_text_context(context)
    assert mask[:, :7].all()
    assert not mask[:, 7:].any()


def test_truncation_precedes_proprio_and_clamps_lengths(monkeypatch):
    driver = _driver()
    model = joint.DualSystemSelfAttnArchitecture()
    model.video_backbone = driver.vb
    model.action_backbone = driver.ab
    model._init_proprio_context({"use_proprioception": True, "state_dim": 3}, text_dim=16)
    context = torch.randn(2, 512, 16)
    proprio = torch.randn(2, 3)

    class Prepared(Exception):
        pass

    def prepare(**kwargs):
        assert kwargs["context"].shape == (2, 129, 16)
        torch.testing.assert_close(kwargs["context"][:, :128], context[:, :128])
        torch.testing.assert_close(kwargs["context"][:, 128], model.proprio_encoder(proprio))
        assert kwargs["context_mask"][:, 128].all()
        assert kwargs["context_mask"][0, :128].all()
        assert kwargs["context_mask"][1, :128].sum() == 7
        torch.testing.assert_close(kwargs["seq_lens"], torch.tensor([128, 7]))
        raise Prepared

    monkeypatch.setattr(driver.vb, "prepare", prepare)
    with pytest.raises(Prepared):
        model(None, None, context=context, seq_lens=torch.tensor([200, 7]), proprio=proprio)


def test_image_conditioned_wan_keeps_fixed_text_boundary(monkeypatch):
    driver = _driver()
    model = joint.DualSystemSelfAttnArchitecture()
    model.video_backbone = driver.vb
    model.action_backbone = driver.ab
    driver.vb.dit.has_image_input = True
    context = torch.randn(2, 512, 16)
    mask = torch.zeros(2, 512, dtype=torch.bool)
    mask[:, :7] = True
    class Prepared(Exception):
        pass
    def prepare(**kwargs):
        assert kwargs['context'].shape[1] == 512
        raise Prepared
    monkeypatch.setattr(driver.vb, 'prepare', prepare)
    monkeypatch.setattr(joint, '_fixed_wan_text_context',
                        lambda *args: pytest.fail('I2V text layout must not be compacted'))
    with pytest.raises(Prepared):
        model(None, None, context=context, context_mask=mask)
