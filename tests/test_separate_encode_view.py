"""Independent camera encoding, local positions, and train/deploy geometry."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.mixture import MixtureDataset
from openwam.dataloader.transforms.multiview import assemble_two_view_layout
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from openwam.model.video_backbone.encoder.vjepa21_src.vision_transformer import (
    VisionTransformer,
)
from openwam.model.video_backbone.wan22_style_backbone import Wan22Style
from tests.dataloader.test_libero import native_libero_bucket
from tests.test_history_context import RecordingViT
from tests.test_vjepa_patch_grid import ROOT, _joint_model, encoder_source
from tests.test_wan22_style_capacity import launcher_config
from tests.test_wan_text_embedding_cache import make_cache


@pytest.mark.parametrize("enabled", [False, True])
def test_single_switch_controls_data_and_legacy_default(enabled):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "dataloader=libero",
                "model/video_backbone/encoder=vjepa21",
                f"model.video_backbone.encoder.seperate_encode_view={str(enabled).lower()}",
            ],
        )
    assert cfg.dataloader.seperate_encode_view is enabled
    obs = ObsPreprocessor.from_cfg(cfg)
    assert (obs.img_height, obs.img_width) == ((224, 448) if enabled else (384, 320))


def test_legacy_skeleton_has_no_separate_view_behavior(encoder_source):
    enc = VJEPA21VideoEncoder.from_skeleton(
        {}, encoder_cfg={}, ckpt_dir=str(encoder_source)
    )
    assert enc.seperate_encode_view is False


@pytest.mark.parametrize("history", [False, True])
def test_reader_decodes_native_views_at_224_and_keeps_temporal_masks(
    native_libero_bucket, monkeypatch, history
):
    calls = []

    def decode(path, indices, h, w, **kwargs):
        camera = Path(path).parent.parent.name
        calls.append((camera, list(indices), h, w, kwargs))
        offset = 100 if camera.endswith("image2") else 0
        return [Image.new("RGB", (w, h), (offset + i,) * 3) for i in indices]

    monkeypatch.setattr(
        "openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", decode
    )
    kwargs = dict(
        dataset_dir=str(native_libero_bucket),
        normalize_mode=None,
        multiview=True,
        history_num_tubelets=int(history),
    )
    old = LiberoDataset(**kwargs)
    new = LiberoDataset(**kwargs, seperate_encode_view=True)
    expected = old[35]
    calls.clear()
    actual = new[35]
    assert len(calls) == 2
    for _, indices, h, w, resample in calls:
        assert (h, w) == (224, 224)
        assert resample == {"resample": Image.Resampling.BILINEAR}
        assert indices == ([31, 35, 39] if history else [35, 39])
    for key in ("action", "action_mask", "proprio", "proprio_mask", "video_mask"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert actual["video"][0].size == (448, 224)
    assert actual["video"][0].getpixel((0, 0)) == (35,) * 3
    assert actual["video"][0].getpixel((224, 0)) == (135,) * 3
    assert actual["video"][-1].getpixel((224, 0)) == (139,) * 3
    if history:
        assert actual["clean_condition_images"][0].getpixel((224, 0)) == (131,) * 3
    mixture = MixtureDataset.from_config(
        {
            "seperate_encode_view": True,
            "datasets": {
                "libero": {
                    "type": "libero",
                    "dataset_dir": str(native_libero_bucket),
                    "normalize_mode": None,
                    "multiview": True,
                }
            },
        }
    )
    assert mixture.get_dataset("libero")._seperate_encode_view


def test_resize_packing_matches_deploy_and_masks_only_missing_view():
    rng = np.random.default_rng(17)
    head = Image.fromarray(rng.integers(0, 256, (256, 320, 3), dtype=np.uint8))
    wrist = Image.fromarray(rng.integers(0, 256, (160, 240, 3), dtype=np.uint8))
    # Reader resizes once during decoding; transport receives those resized images.
    train = assemble_two_view_layout(
        head.resize((224, 224), Image.Resampling.BILINEAR),
        wrist.resize((224, 224), Image.Resampling.BILINEAR),
    )
    pre = ObsPreprocessor(
        multiview=True,
        camera_layout=["head", "wrist", None],
        img_height=384,
        img_width=320,
        seperate_encode_view=True,
    )
    deploy = pre.preprocess(
        {"images": {"head_camera": head, "left_wrist_camera": wrist}}
    )["image"]
    np.testing.assert_array_equal(np.asarray(train), np.asarray(deploy))
    image, mask = assemble_two_view_layout(head, None, return_missing_mask=True)
    assert not np.asarray(image)[:, 224:].any()
    assert not np.asarray(mask)[:, :224].any()
    assert (np.asarray(mask)[:, 224:] == 255).all()
    with pytest.raises(ValueError, match="exactly two"):
        ObsPreprocessor(
            multiview=True,
            camera_layout=["head", "left", "right"],
            img_height=384,
            img_width=320,
            seperate_encode_view=True,
        )


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("pooling", [False, True])
@pytest.mark.parametrize("mode", ["video", "mixed"])
def test_each_context_and_teacher_call_batches_views_without_mixing(
    history, pooling, mode
):
    vit = RecordingViT()
    enc = VJEPA21VideoEncoder(
        vit,
        embed_dim=8,
        variant="test",
        seperate_encode_view=True,
        history_num_tubelets=int(history),
        use_avg_t_pooling=pooling,
        vjepa21_forward=mode,
    )
    frames = 10 if history else 9
    packed = torch.zeros(2, 3, frames, 224, 448)
    for b in range(2):
        for v in range(2):
            packed[b, :, :, :, v * 224 : (v + 1) * 224] = 10 * b + v
    z = enc.batch_encode(packed)
    assert z.shape == (2, 8, 3 if pooling else 5, 14, 28)
    assert len(vit.inputs) == 2
    for x in vit.inputs:
        assert x.shape[0] == 4 and x.shape[-2:] == (224, 224)
        assert x[:, 0, 0, 0, 0].tolist() == [0, 1, 10, 11]
    with pytest.raises(ValueError, match="224x448"):
        enc.batch_encode(torch.zeros(1, 3, frames, 384, 320))


def test_real_vit_matches_independent_calls_and_other_view_cannot_leak():
    torch.manual_seed(7)
    vit = VisionTransformer(
        img_size=(224, 224),
        patch_size=16,
        num_frames=2,
        tubelet_size=2,
        embed_dim=48,
        depth=12,
        num_heads=4,
        mlp_ratio=2,
        use_rope=True,
        interpolate_rope=True,
        img_temporal_dim_size=1,
    ).eval()
    enc = VJEPA21VideoEncoder(
        vit, history_num_tubelets=0, embed_dim=48, variant="tiny", seperate_encode_view=True
    ).eval()
    packed = torch.randn(2, 3, 2, 224, 448)
    with torch.no_grad():
        actual = enc._vit_grid(packed)
        refs = [
            vit(packed[..., v * 224 : (v + 1) * 224])
            .transpose(1, 2)
            .reshape(2, 48, 1, 14, 14)
            for v in range(2)
        ]
        torch.testing.assert_close(
            actual, torch.cat(refs, dim=-1), atol=2e-6, rtol=2e-5
        )
        changed = packed.clone()
        changed[..., 224:] = torch.randn_like(changed[..., 224:])
        perturbed = enc._vit_grid(changed)
    torch.testing.assert_close(actual[..., :14], perturbed[..., :14], atol=0, rtol=0)
    assert not torch.allclose(actual[..., 14:], perturbed[..., 14:])


def _backbone(tmp_path, encoder_source, grid, history=True):
    cfg = launcher_config()
    cfg.model.video_backbone.encoder.model_path = str(encoder_source)
    cfg.model.video_backbone.encoder.seperate_encode_view = True
    cfg.model.video_backbone.encoder.use_wan_style_patch_grid = grid
    cfg.model.video_backbone.encoder.use_avg_t_pooling = False
    cfg.model.video_backbone.encoder.history_num_tubelets = int(history)
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in {
        "dim": 96,
        "ffn_dim": 192,
        "num_heads": 4,
        "num_layers": 1,
        "freq_dim": 32,
    }.items():
        cfg.model.video_backbone.dit_config[key] = value
    enc = VJEPA21VideoEncoder(
        RecordingViT(),
        embed_dim=8,
        variant="tiny",
        seperate_encode_view=True,
        use_wan_style_patch_grid=grid,
        history_num_tubelets=int(history),
        use_avg_t_pooling=False,
    )
    return Wan22Style.from_pretrained(cfg, external_encoder=enc, device="cpu")


@pytest.mark.parametrize("grid", [False, True])
def test_local_rope_and_view_embedding_follow_row_major_layout(
    tmp_path, encoder_source, grid
):
    vb = _backbone(tmp_path, encoder_source, grid)
    vb.set_dtype_device(torch.float32, torch.device("cpu"))
    latents = torch.randn(1, 8, 3, 14, 28)
    with torch.no_grad():
        vb.dit.view_embedding.weight[0].fill_(2)
        vb.dit.view_embedding.weight[1].fill_(5)
    state = vb.prepare(
        latents=latents, context=torch.randn(1, 3, 4096), timestep=torch.tensor([300.0])
    )
    h, w = (7, 7) if grid else (14, 14)
    rope = state.rope_freqs.reshape(3, h, 2 * w, -1)
    torch.testing.assert_close(rope[:, :, :w], rope[:, :, w:], atol=0, rtol=0)
    expected = torch.cat(
        [
            vb.dit.freqs[0][:3, None, None].expand(3, h, w, -1),
            vb.dit.freqs[1][None, :h, None].expand(3, h, w, -1),
            vb.dit.freqs[2][None, None, :w].expand(3, h, w, -1),
        ],
        dim=-1,
    )
    torch.testing.assert_close(rope[:, :, :w], expected, atol=0, rtol=0)
    hidden = state.hidden_states.reshape(1, 3, h, 2 * w, 96)
    projected = vb.dit.patchify(latents).permute(0, 2, 3, 4, 1)
    torch.testing.assert_close(hidden[:, :, :, :w], projected[:, :, :, :w] + 2)
    torch.testing.assert_close(hidden[:, :, :, w:], projected[:, :, :, w:] + 5)
    # 2x2 patch windows never straddle the view boundary.
    changed = latents.clone()
    changed[..., 14:] += 1
    torch.testing.assert_close(
        vb.dit.patchify(changed)[..., :w], vb.dit.patchify(latents)[..., :w]
    )


@pytest.mark.parametrize("grid", [False, True])
def test_training_and_deployment_use_same_context_and_noise_geometry(
    tmp_path, encoder_source, grid
):
    vb = _backbone(tmp_path, encoder_source, grid)
    model = _joint_model(vb)
    frames = [
        assemble_two_view_layout(
            Image.new("RGB", (224, 224), (i * 10,) * 3),
            Image.new("RGB", (224, 224), (100 + i * 10,) * 3),
        )
        for i in range(9)
    ]
    history = assemble_two_view_layout(
        Image.new("RGB", (224, 224), (200,) * 3),
        Image.new("RGB", (224, 224), (10,) * 3),
    )
    sample = dict(
        video=frames,
        clean_condition_images=[history, frames[0]],
        prompt="open drawer",
        action=torch.zeros(32, 80),
        action_mask=torch.ones(32, 80, dtype=torch.bool),
        video_mask=torch.tensor([True] * 5 + [False] * 4),
    )
    train = model.prepare_inputs([sample])
    assert train["input_latents"].shape == (1, 8, 5, 14, 28)
    assert train["video_is_pad"].tolist() == [[False, False, True, True]]
    train_pixels = vb.video_encoder._m.inputs[0].clone()
    deploy = vb.preprocess_input_for_inference(
        prompt="open drawer",
        clean_condition_images=[history, frames[0]],
        height=384,
        width=320,
        num_frames=9,
        num_inference_steps=2,
    )
    assert (deploy["height"], deploy["width"]) == (224, 448)
    assert deploy["latents"].shape == train["input_latents"].shape
    torch.testing.assert_close(
        vb.video_encoder._m.inputs[-1], train_pixels, atol=0, rtol=0
    )
    torch.testing.assert_close(
        deploy["clean_condition_latents"], train["input_latents"][:, :, :1]
    )
