"""Frame-shared modulation must match the original per-token joint blocks."""
import pytest
import torch

from tests.test_mot_first_frame_attention import _driver


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('frames,clean,spatial', [(1, 0, 4), (3, 1, 4), (5, 2, 8)])
def test_compact_joint_output_and_all_gradients(checkpoint, frames, clean, spatial):
    torch.manual_seed(77)
    driver = _driver()
    vb, ab = driver.vb, driver.ab

    def run(compact):
        torch.manual_seed(32)
        vb.zero_grad(set_to_none=True)
        ab.zero_grad(set_to_none=True)
        latents = torch.randn(2, 4, frames, spatial, spatial, requires_grad=True)
        ts = torch.tensor([121., 634.], requires_grad=True)
        context = torch.randn(2, 4, 16)
        vs = vb.prepare(latents=latents, timestep=ts, context=context,
                        num_clean_condition_latents=clean, fuse_vae_embedding_in_latents=True)
        acs = ab.prepare_state(torch.randn(2, 3, 7), torch.tensor([222., 431.]), context=context)
        if compact:
            vb.compact_time_modulation_for_joint(vs)
            assert vs.time_mod.shape == (2, frames, 6, vb.dim)
        vs, acs = driver.run_joint_loop(vs, acs, use_gradient_checkpointing=checkpoint)
        # Include the unchanged final head, not just block hidden states.
        output = torch.cat([vb.finalize(vs).flatten(), ab.extract_prediction(acs).flatten()])
        (output * torch.randn_like(output)).sum().backward()
        grads = {prefix + name: p.grad.detach().clone()
                 for prefix, module in [('v.', vb), ('a.', ab)]
                 for name, p in module.named_parameters() if p.grad is not None}
        grads.update(latents=latents.grad.clone(), timestep=ts.grad.clone())
        return output.detach(), grads

    expected, ref_grads = run(False)
    actual, grads = run(True)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], atol=5e-5, rtol=5e-5, msg=name)
