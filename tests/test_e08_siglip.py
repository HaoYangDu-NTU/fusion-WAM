import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from openwam.experiments import resolve_experiment
from openwam.model.video_backbone.encoder.siglip import SiglipVideoEncoder
from openwam.dataloader.libero_native_stereo import NativeStereoLiberoDataset
from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.transforms.native_stereo import assemble_native_stereo
from openwam.deploy.obs_preprocess import ObsPreprocessor
from tests.dataloader.test_libero import native_libero_bucket


class SpatialSiglip(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(hidden_size=1024, image_size=256, patch_size=16)
        self.use_head = False

    def forward(self, pixel_values):
        patches = torch.nn.functional.avg_pool2d(pixel_values.float(), 16).mean(1).flatten(1)
        channels = torch.arange(1024, device=pixel_values.device)/1024
        return SimpleNamespace(last_hidden_state=patches[:, :, None]+channels[None, None, :])


def test_siglip_patch_order_freezing_and_future_isolation():
    encoder = SiglipVideoEncoder(SpatialSiglip()).train()
    assert not encoder._m.training and not any(p.requires_grad for p in encoder.parameters())
    video = torch.zeros(2, 3, 10, 256, 512)
    for batch in range(2):
        for t in range(10):
            video[batch, :, t, :, :256] = 100*batch+10*t
            video[batch, :, t, :, 256:] = 100*batch+10*t+1
    raw = encoder._encode_frames(video)
    assert raw.shape == (2, 1024, 10, 16, 32)
    for b in range(2):
        for t in range(10):
            assert raw[b, 0, t, 0, 0] == 100*b+10*t
            assert raw[b, 0, t, 0, 16] == 100*b+10*t+1
    original = encoder.batch_encode(video)
    video[:, :, 2:] += 7
    torch.testing.assert_close(original[:, :, :2], encoder.batch_encode(video)[:, :, :2], rtol=0, atol=0)
    projection = encoder.build_dit_input_proj(96)
    assert projection(original, num_clean_condition_latents=2).shape == (2, 96, 10, 8, 16)


def test_native_resize_deploy_and_official_processor_match():
    from transformers import AutoImageProcessor
    cfg = resolve_experiment("E08")[0]
    rng = np.random.default_rng(42)
    front = Image.fromarray(rng.integers(0, 256, (320, 400, 3), dtype=np.uint8))
    wrist = Image.fromarray(rng.integers(0, 256, (180, 240, 3), dtype=np.uint8))
    packed = assemble_native_stereo(front, wrist)
    preprocessor = ObsPreprocessor.from_cfg(cfg)
    deployed = preprocessor.preprocess(dict(images=dict(head_camera=front, left_wrist_camera=wrist),
                                            prompt="pick", state=[0.0]*10))["image"]
    assert preprocessor.img_height == 256 and preprocessor.img_width == 512
    assert np.array_equal(np.asarray(packed), np.asarray(deployed))
    encoder = SiglipVideoEncoder(SpatialSiglip())
    processed = encoder.preprocess_video([packed])[0, :, 0]
    reference = AutoImageProcessor.from_pretrained(
        cfg.model.video_backbone.encoder.model_path, local_files_only=True, use_fast=False)(
        images=[front, wrist], return_tensors="pt")["pixel_values"]
    torch.testing.assert_close(processed.to(torch.bfloat16),
        torch.cat([reference[0], reference[1]], dim=-1).to(torch.bfloat16), rtol=0, atol=0)


def test_native_reader_keeps_action_history_masks_and_decodes_once(native_libero_bucket, monkeypatch):
    calls = []
    def decode(path, indices, h, w, **kwargs):
        camera = Path(path).parent.parent.name
        offset = 100 if camera.endswith("image2") else 0
        calls.append((list(indices), h, w, kwargs))
        return [Image.new("RGB", (w, h), (offset+i,)*3) for i in indices]
    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", decode)
    monkeypatch.setattr("openwam.dataloader.libero_native_stereo._decode_video_frames", decode)
    options = dict(dataset_dir=str(native_libero_bucket), normalize_mode=None, multiview=True,
                   seperate_encode_view=True, history_num_tubelets=1)
    base, native = LiberoDataset(**options), NativeStereoLiberoDataset(**options)
    old = base[35]
    calls.clear()
    current = native[35]
    assert len(calls) == 2
    assert all((h, w, extra) == (256, 256, dict(resample=Image.Resampling.BICUBIC)) for _, h, w, extra in calls)
    assert all(indices == [31, 35, 39] for indices, *_ in calls)
    for key in ("action", "action_mask", "video_mask", "proprio", "proprio_mask"):
        torch.testing.assert_close(current[key], old[key], rtol=0, atol=0)
    assert current["prompt"] == old["prompt"]
    assert current["video"][0].size == (512, 256)
    assert current["clean_condition_images"][0].getpixel((256, 0)) == (131,)*3


def test_e08_keeps_e07_budget_model_except_native_encoder_and_geometry():
    base = OmegaConf.to_container(resolve_experiment("E07-Cache")[0], resolve=True)
    current, profile, _ = resolve_experiment("E08-Cache")
    current = OmegaConf.to_container(current, resolve=True)
    assert profile["geometry"]["tokens"] == 1280
    assert current["model"]["architecture"] == base["model"]["architecture"]
    assert current["model"]["action_backbone"] == base["model"]["action_backbone"]
    assert current["model"]["video_backbone"]["dit_config"] == base["model"]["video_backbone"]["dit_config"]
    for value in (base, current):
        value["training"].pop("output_path")
    assert current["training"] == base["training"]
    assert current["dataloader"]["native_view_size"] == 256
    assert current["training"]["lambda_video"] == 1


def test_new_cache_dispatch_preserves_old_identity(monkeypatch):
    from openwam.dataloader import spatial_frame_cache_specs as specs
    sentinel = {"unchanged": True}
    monkeypatch.setattr(specs, "existing_identity", lambda *args: sentinel)
    assert specs.frame_cache_identity({}, {"name": "dinov3"}, None) is sentinel


def test_e08_report_does_not_replace_e07(tmp_path, monkeypatch):
    import scripts.e07_train_eval_queue as queue
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    destination = tmp_path/"queue"
    destination.mkdir()
    (tmp_path/"result.md").write_text("<!-- E07_AUTO_RESULT_BEGIN -->E07 existing<!-- E07_AUTO_RESULT_END -->\n| E08 | pending |\n")
    result = dict(successes=1900, overall=.95, suites={suite: .95 for suite in queue.SUITES}, summary="summary.json")
    queue.write_report(destination, result, "E08")
    queue.write_report(destination, result, "E08")
    text = (tmp_path/"result.md").read_text()
    assert "E07 existing" in text and text.count("<!-- E08_AUTO_RESULT_BEGIN -->") == 1
    assert "SigLIP-L/16" in text and "1280" in text and "95.00%" in text
