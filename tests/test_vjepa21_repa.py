"""Independent teacher fusion, native output preservation and one-pass extraction."""

import pytest
import torch

from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from openwam.model.video_backbone.encoder.vjepa21_src.vision_transformer import VisionTransformer
from tests.test_vjepa21_fusion import GridViT


class RepaGridViT(GridViT):
    def iter_intermediate_layers(self, pixels, layer_indices, *, include_final_output=False):
        self.intermediate_calls += 1
        for i, grid in enumerate(self.raw_layers(pixels)):
            if i in layer_indices:
                yield grid
        if include_final_output:
            yield self.norm(grid)


@pytest.mark.parametrize("main_method", ["last_layer", "mean", "mean_plus_global"])
@pytest.mark.parametrize("target_method", ["last_layer", "mean", "mean_plus_global"])
@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("views", [False, True])
def test_teacher_selection_keeps_main_latents_and_uses_two_total_passes(main_method, target_method, pool, views):
    torch.manual_seed(29)
    vit = RepaGridViT()
    encoder = VJEPA21VideoEncoder(
        vit, embed_dim=8, variant="test", history_num_tubelets=1,
        use_avg_t_pooling=pool, seperate_encode_view=views,
        future_fusion_method=main_method, fusion_layers=[0, 2],
    ).eval().requires_grad_(False)
    height, width = (224, 448) if views else (32, 48)
    pixels = torch.randn(1, 3, 10, height, width)
    expected_main = encoder.batch_encode(pixels)
    # Independent legacy encode is the teacher oracle, with different selected layers.
    encoder._future_fusion_method, encoder._fusion_layers = target_method, (1, 3)
    expected_teacher = encoder.batch_encode(pixels)[:, :, 1:]
    encoder._future_fusion_method, encoder._fusion_layers = main_method, (0, 2)
    vit.native_calls = vit.intermediate_calls = 0
    rng = torch.get_rng_state()
    actual, teacher = encoder.batch_encode_with_repa_target(
        pixels, target_fusion_method=target_method, target_fusion_layers=[1, 3],
    )
    torch.testing.assert_close(actual, expected_main, rtol=0, atol=0)
    torch.testing.assert_close(teacher, expected_teacher, rtol=0, atol=0)
    assert vit.native_calls + vit.intermediate_calls == 2
    assert torch.equal(torch.get_rng_state(), rng)
    assert not actual.requires_grad and not teacher.requires_grad


@pytest.mark.parametrize("selected", [(), (0, 4, 11)])
def test_real_vit_appends_native_final_norm_in_same_pass(selected):
    vit = VisionTransformer(
        img_size=32, patch_size=16, num_frames=6, tubelet_size=2,
        embed_dim=48, depth=12, num_heads=2, use_rope=True, img_temporal_dim_size=1,
    ).eval().requires_grad_(False)
    pixels = torch.randn(1, 3, 6, 32, 48)
    expected = vit(pixels)
    raw = list(vit.iter_intermediate_layers(pixels, selected))
    calls = []
    handle = vit.blocks[0].register_forward_hook(lambda *args: calls.append(1))
    outputs = list(vit.iter_intermediate_layers(pixels, selected, include_final_output=True))
    handle.remove()
    assert len(calls) == 1 and len(outputs) == len(selected) + 1
    for actual, target in zip(outputs[:-1], raw):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    torch.testing.assert_close(outputs[-1].flatten(1, 3), expected, rtol=0, atol=0)


def test_auxiliary_layers_validate_even_when_main_uses_last_layer():
    encoder = VJEPA21VideoEncoder(RepaGridViT(), history_num_tubelets=0, embed_dim=8, variant="test")
    assert encoder.resolve_repa_target_layers("last_layer", [39]) == ()
    assert encoder.resolve_repa_target_layers("mean", "all_layers") == (0, 1, 2, 3)
    for method, layers in [("typo", [1]), ("mean", [39]), ("mean", [2, 1]), ("mean", [])]:
        with pytest.raises(ValueError):
            encoder.resolve_repa_target_layers(method, layers)
