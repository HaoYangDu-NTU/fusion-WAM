"""Episode-local history sampling and the V-JEPA context contract."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from openwam.history import condition_frame_offsets, history_from_config, validate_history_num_tubelets
from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.mixture import MixtureDataset
from openwam.deploy.policy import WAMPolicy
from openwam.model.video_backbone.encoder.registry import build_video_encoder
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
from openwam.model.video_backbone.wan.conditioning import finalize_ti2v_clean_condition_latents
from openwam.model.video_backbone.wan22_style_backbone import Wan22Style
from tests.dataloader.test_libero import native_libero_bucket
from tests.test_vjepa_patch_grid import ROOT, _joint_model, encoder_source
from tests.test_vjepa_svae import _MockVJEPAViT
from tests.test_wan22_style_capacity import launcher_config
from tests.test_wan_text_embedding_cache import make_cache


@pytest.mark.parametrize("enabled", [0, 1, 2, 3])
def test_one_encoder_override_controls_sampling(enabled):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "dataloader=libero", "model/video_backbone/encoder=vjepa21",
            f"model.video_backbone.encoder.history_num_tubelets={enabled}",
        ])
    assert cfg.model.video_backbone.encoder.history_num_tubelets == enabled
    assert cfg.dataloader.history_num_tubelets == enabled


@pytest.mark.parametrize("enabled", [0, 1, 2, 3])
def test_mixture_propagates_history_to_reader(native_libero_bucket, enabled):
    mixture = MixtureDataset.from_config({
        "history_num_tubelets": enabled,
        "datasets": {"libero": {"type": "libero", "dataset_dir": str(native_libero_bucket),
                                "normalize_mode": None, "multiview": True}},
    })
    assert mixture.get_dataset("libero")._history_num_tubelets == enabled


@pytest.mark.parametrize("num_history", [1, 2, 3])
@pytest.mark.parametrize("episode,step", [(0, 0), (0, 1), (0, 3), (0, 4), (0, 8), (0, 39), (1, 0), (1, 3), (1, 8)])
def test_history_clamps_to_episode_start_and_preserves_future(native_libero_bucket, monkeypatch, episode, step, num_history):
    root = native_libero_bucket
    metadata = root / "meta/episodes/chunk-000.parquet"
    episodes = pd.read_parquet(metadata)
    second = episodes.iloc[0].copy()
    second["episode_index"], second["dataset_from_index"] = 1, 40
    pd.concat([episodes, second.to_frame().T], ignore_index=True).to_parquet(metadata)
    data_path = root / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(data_path)
    pd.concat([data, data], ignore_index=True).to_parquet(data_path)
    decoded = []

    def decode(path, frame_indices, h, w):
        decoded.append(tuple(frame_indices))
        return [Image.new("RGB", (w, h), (i, i, i)) for i in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", decode)
    kwargs = dict(dataset_dir=str(root), normalize_mode=None, multiview=True, height=384, width=320)
    baseline = LiberoDataset(**kwargs)
    history = LiberoDataset(**kwargs, history_num_tubelets=num_history)
    # Both episodes share one MP4: the clamp must happen before its file offset.
    index = int(history._cum_n_starts[episode]) + step
    expected = baseline[index]
    decoded.clear()
    sample = history[index]
    file_start = episode * 40
    pair = [file_start + max(step + offset, 0) for offset in condition_frame_offsets(num_history)]
    assert [frame.getpixel((10, 10))[0] for frame in sample["clean_condition_images"]] == pair
    assert all(indices[:2 * num_history] == tuple(pair) for indices in decoded)
    assert len(sample["video"]) == 9
    for actual, original in zip(sample["video"], expected["video"]):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(original))
    for key in ("action", "action_mask", "proprio", "proprio_mask", "video_mask"):
        torch.testing.assert_close(sample[key], expected[key], rtol=0, atol=0)
    assert sample["prompt"] == expected["prompt"]


def test_history_and_future_share_augmentation(native_libero_bucket, monkeypatch):
    def decode(path, indices, h, w):
        return [Image.new("RGB", (w, h), (i, i, i)) for i in indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", decode)
    dataset = LiberoDataset(dataset_dir=str(native_libero_bucket), normalize_mode=None,
                            multiview=True, height=384, width=320, history_num_tubelets=1)
    seen = []

    def augment(sample):
        seen.append([frame.getpixel((10, 10))[0] for frame in sample["video"]])
        assert len(sample["video_jitter_exclusion_masks"]) == 10
        return {**sample, "video": [Image.new("RGB", frame.size, (100 + i,) * 3)
                                    for i, frame in enumerate(sample["video"])]}

    dataset._color_jitter = SimpleNamespace(apply=augment)
    sample = dataset[8]
    assert seen == [[4, 8, 12, 16, 20, 24, 28, 32, 36, 36]]
    assert [frame.getpixel((10, 10))[0] for frame in sample["clean_condition_images"]] == [100, 101]
    assert sample["video"][0].getpixel((10, 10))[0] == 101


class RecordingViT(_MockVJEPAViT):
    def __init__(self):
        super().__init__(8)
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return super().forward(x)


@pytest.mark.parametrize("history", [0, 1, 2, 3])
@pytest.mark.parametrize("pooling", [False, True])
@pytest.mark.parametrize("mode", ["video", "mixed"])
def test_encoder_uses_clean_context_and_full_teacher(history, pooling, mode):
    vit = RecordingViT()
    encoder = VJEPA21VideoEncoder(vit, embed_dim=8, variant="test", vjepa21_forward=mode,
                                 history_num_tubelets=history, use_avg_t_pooling=pooling)
    offsets = condition_frame_offsets(history) + list(range(4, 33, 4))
    video = torch.tensor(offsets, dtype=torch.float32).reshape(1, 1, -1, 1, 1).expand(1, 3, -1, 32, 32)
    output = encoder.batch_encode(video)
    expected_context = condition_frame_offsets(history) if history else ([0] if mode == "mixed" else [0, 0])
    assert vit.inputs[0][0, 0, :, 0, 0].tolist() == expected_context
    assert vit.inputs[1][0, 0, :, 0, 0].tolist() == (offsets if history else [0, *offsets])
    assert output.shape == (1, 8, max(1, history) + (2 if pooling else 4), 2, 2)
    vit.inputs.clear()
    context_only = video[:, :, :max(1, 2 * history)]
    assert encoder.batch_encode(context_only).shape == (1, 8, max(1, history), 2, 2)
    assert len(vit.inputs) == 1


def test_enabled_encoder_requires_history_and_legacy_skeleton_defaults_off(encoder_source):
    encoder = VJEPA21VideoEncoder(RecordingViT(), embed_dim=8, variant="test", history_num_tubelets=1)
    with pytest.raises(ValueError, match="clean condition images"):
        encoder.batch_encode(torch.zeros(1, 3, 1, 32, 32))
    restored = VJEPA21VideoEncoder.from_skeleton({}, ckpt_dir=str(encoder_source), encoder_cfg={})
    assert restored.history_num_tubelets == 0


@pytest.mark.parametrize("history_count", [1, 2, 3])
def test_training_preprocessing_keeps_padding_and_matches_deploy_context(tmp_path, encoder_source, history_count):
    cfg = launcher_config()
    cfg.model.video_backbone.encoder.model_path = str(encoder_source)
    cfg.model.video_backbone.encoder.history_num_tubelets = history_count
    cfg.model.video_backbone.text_embedding_dir = str(make_cache(tmp_path))
    for key, value in {"dim": 96, "ffn_dim": 192, "num_heads": 4, "num_layers": 1, "freq_dim": 32}.items():
        cfg.model.video_backbone.dit_config[key] = value
    encoder = build_video_encoder(cfg.model.video_backbone.encoder)
    encoder._m = RecordingViT()
    backbone = Wan22Style.from_pretrained(cfg, external_encoder=encoder, device="cpu")
    model = _joint_model(backbone)
    frames = [Image.new("RGB", (64, 64), (i * 10,) * 3) for i in range(9)]
    condition = [Image.new("RGB", (64, 64), (100 + i,) * 3) for i in range(2 * history_count - 1)] + [frames[0]]
    sample = dict(video=frames, clean_condition_images=condition, prompt="open drawer",
                  action=torch.zeros(32, 80), action_mask=torch.ones(32, 80, dtype=torch.bool),
                  video_mask=torch.tensor([True] * 5 + [False] * 4))
    inputs = model.prepare_inputs([sample])
    assert inputs["num_frames"] == 9
    assert inputs["input_latents"].shape == (1, 8, history_count + 2, 4, 4)
    assert inputs["video_is_pad"].tolist() == [[False, True]]
    assert inputs["actions"].shape == (1, 32, 80)
    assert encoder._m.inputs[0].shape[2] == 2 * history_count
    assert encoder._m.inputs[1].shape[2] == 2 * history_count + 8
    train_context_pixels = encoder._m.inputs[0].clone()
    deploy = {}
    finalize_ti2v_clean_condition_latents(deploy, condition, is_ti2v=True,
                                     encoder=encoder, vae=None, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(encoder._m.inputs[-1], train_context_pixels, rtol=0, atol=0)
    torch.testing.assert_close(deploy["clean_condition_latents"], inputs["input_latents"][:, :, :history_count])
    with pytest.raises(ValueError, match="reference frames"):
        backbone.preprocess_input_for_train(frames=[frames], text=["open drawer"], ref_images=[[frames[0]]])


@pytest.mark.parametrize("enabled", [0, 1, 2, 3])
def test_policy_records_every_control_step_and_resets(enabled):
    references = []

    def generate(conditions):
        references.append(conditions["clean_condition_images"])
        return {"actions": np.zeros((6, 7), dtype=np.float32)}

    encoder = SimpleNamespace(history_num_tubelets=enabled)
    engine = SimpleNamespace(generate=generate,
                             architecture=SimpleNamespace(video_backbone=SimpleNamespace(video_encoder=encoder)))
    policy = WAMPolicy(engine, SimpleNamespace(), execution_config={"mode": "sync", "inference_horizon": 6})
    for step in range(13):
        policy.predict_action({"image": step, "prompt": "test"})
    assert references == [[max(0, step + offset) for offset in condition_frame_offsets(enabled)]
                          for step in (0, 6, 12)]
    policy.reset()
    policy.predict_action({"image": 100})
    assert references[-1] == [100] * max(1, 2 * enabled)


@pytest.mark.parametrize("config,expected", [
    ({}, 0), ({"add_one_history_context": False}, 0), ({"add_one_history_context": True}, 1),
    ({"history_num_tubelets": 0}, 0), ({"history_num_tubelets": 2}, 2),
    ({"history_num_tubelets": 1, "add_one_history_context": True}, 1),
])
def test_saved_history_config_migration(config, expected, encoder_source):
    assert history_from_config(config) == expected
    restored = VJEPA21VideoEncoder.from_skeleton({}, ckpt_dir=str(encoder_source), encoder_cfg=config)
    assert restored.history_num_tubelets == expected


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_invalid_history_count_is_rejected(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_history_num_tubelets(value)


def test_new_encoder_default_and_conflicting_legacy_config():
    assert VJEPA21VideoEncoder(RecordingViT(), embed_dim=8, variant="test").history_num_tubelets == 1
    with pytest.raises(ValueError, match="Conflicting"):
        history_from_config({"history_num_tubelets": 2, "add_one_history_context": True})


@pytest.mark.parametrize("history", [0, 1, 2, 3])
def test_teacher_uses_full_history_without_leaking_future_into_clean(history):
    from tests.test_vjepa21_repa import RepaGridViT

    class ContextualViT(RepaGridViT):
        def forward(self, x):
            # Real JEPA self-attention lets each tubelet see the whole input clip.
            grid = super().forward(x)
            clip_mean = x.mean((1, 2, 3, 4)).view(-1, 1, 1)
            return grid + clip_mean * torch.arange(8).square().view(1, 1, -1)

    encoder = VJEPA21VideoEncoder(ContextualViT(), embed_dim=8, variant="test",
                                 history_num_tubelets=history)
    count = max(1, 2 * history)
    n_clean = max(1, history)
    torch.manual_seed(21)
    video = torch.randn(1, 3, count + 4, 32, 32)
    base = encoder.batch_encode(video)
    different_future = video.clone()
    different_future[:, :, count:] += 2
    changed = encoder.batch_encode(different_future)
    torch.testing.assert_close(base[:, :, :n_clean], changed[:, :, :n_clean], rtol=0, atol=0)
    assert not torch.allclose(base[:, :, n_clean:], changed[:, :, n_clean:])
    different_history = video.clone()
    different_history[:, :, :1] += 2
    changed = encoder.batch_encode(different_history)
    assert not torch.allclose(base[:, :, :n_clean], changed[:, :, :n_clean])
    assert not torch.allclose(base[:, :, n_clean:], changed[:, :, n_clean:])


@pytest.mark.parametrize("history", [2, 3])
def test_real_jepa_long_clip_keeps_clean_prefix_independent(history):
    from openwam.model.video_backbone.encoder.vjepa21_src.vision_transformer import VisionTransformer

    torch.manual_seed(29)
    vit = VisionTransformer(img_size=32, patch_size=16, num_frames=2 * history + 4,
                            tubelet_size=2, embed_dim=48, depth=12, num_heads=2,
                            use_rope=True, img_temporal_dim_size=1)
    encoder = VJEPA21VideoEncoder(vit, embed_dim=48, variant="test", use_avg_t_pooling=False,
                                 history_num_tubelets=history).eval().requires_grad_(False)
    pixels = torch.randn(1, 3, 2 * history + 4, 32, 32)
    encoded = encoder.batch_encode(pixels)
    assert encoded.shape == (1, 48, history + 2, 2, 2)
    torch.testing.assert_close(encoded[:, :, :history], encoder.batch_encode(pixels[:, :, :2 * history]),
                               rtol=0, atol=0)
    changed = pixels.clone()
    changed[:, :, 0] += 3
    assert not torch.allclose(encoded[:, :, history:], encoder.batch_encode(changed)[:, :, history:])


@pytest.mark.parametrize("history,length", [(0, 3), (1, 4), (2, 2), (2, 8)])
def test_native_deploy_rejects_incorrect_condition_length(history, length):
    encoder = VJEPA21VideoEncoder(RecordingViT(), embed_dim=8, variant="test", history_num_tubelets=history)
    with pytest.raises(ValueError, match="clean_condition_images"):
        finalize_ti2v_clean_condition_latents({}, [Image.new("RGB", (32, 32))] * length,
                                             is_ti2v=True, encoder=encoder, vae=None,
                                             dtype=torch.float32, device="cpu")


@pytest.mark.parametrize("history", [1, 2, 3])
def test_clean_attention_prefix_and_legacy_mode_alias(history):
    from tests.test_mot_first_frame_attention import _driver

    driver = _driver()
    driver.vb.video_attention_mask_mode = "first_frame_causal"
    assert driver.vb.video_attention_mask_mode == "clean_condition_causal"
    mask = driver._build_attention_mask(s_video=(history + 2) * 4, s_action=3,
                                        video_tokens_per_frame=4, device=torch.device("cpu"),
                                        num_clean_condition_latents=history)
    assert mask[:history * 4, :history * 4].all()
    assert not mask[:history * 4, history * 4:].any()
    assert mask[history * 4:].all()


@pytest.mark.parametrize("history", [0, 1, 2])
def test_share_dir_launcher_history_parameter(history):
    from tests.test_input_bottleneck import launcher_config as resolve_launcher

    cfg = resolve_launcher("train_libero_vjepa_1_3b_dev_share_dir.sh", {"HISTORY_NUM_TUBELETS": str(history)})
    assert cfg.model.video_backbone.encoder.history_num_tubelets == history
    assert cfg.dataloader.history_num_tubelets == history
