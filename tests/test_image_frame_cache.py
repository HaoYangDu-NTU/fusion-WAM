"""Cache bypass, temporal boundaries, worker reads and unchanged M29 gradients."""
import copy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from openwam.dataloader.image_frame_cache import ImageFrameCache, ImageFrameCacheCollator, window_frame_indices
from openwam.experiments import resolve_experiment
from openwam.model import build_architecture, resolve_architecture_config
from openwam.model.video_backbone.encoder.ijepa import IJEPAImageVideoEncoder
from tests.test_exp_dinov3_native import frames
from tests.test_ijepa_encoder import _MockIJEPA
from tests.test_wan_text_embedding_cache import make_cache


def cache_files(path, values, shard_size=4):
    path.mkdir(exist_ok=True)
    first = next(iter(values.values()))
    (path / "manifest.json").write_text(json.dumps(dict(length=max(values)+1, shape=list(first.shape),
        shard_size=shard_size, dtype="bfloat16")))
    for shard in {key // shard_size for key in values}:
        save_file({str(key): value for key, value in values.items() if key // shard_size == shard},
                  path / f"shard_{shard:06d}.safetensors")


class MetadataSamples(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return dict(image_frame_indices=[index, index, (index+1) % 4], video=[], prompt=str(index))


def test_mmap_batch_order_dedup_and_worker_prefetch(tmp_path):
    values = {i: torch.full((2, 2, 4), i, dtype=torch.bfloat16) for i in range(4)}
    cache_files(tmp_path, values, shard_size=2)
    cache = ImageFrameCache(tmp_path, max_open_shards=1)
    cache.require_complete()
    restored = cache.read_many([3, 0, 3, 2])
    for key in (3, 0, 2):
        torch.testing.assert_close(restored[key], values[key], rtol=0, atol=0)
    assert len(cache._handles) == 1
    restored[3].zero_()
    torch.testing.assert_close(cache.read_many([3])[3], values[3], rtol=0, atol=0)
    clone = pickle.loads(pickle.dumps(cache))
    assert not clone._handles
    loader = torch.utils.data.DataLoader(MetadataSamples(), batch_size=2, num_workers=2,
        multiprocessing_context="spawn", prefetch_factor=2, collate_fn=ImageFrameCacheCollator(tmp_path))
    samples = [sample for batch in loader for sample in batch]
    for i, sample in enumerate(samples):
        actual = sample["image_frame_features"]
        torch.testing.assert_close(actual[:, 0], values[i], rtol=0, atol=0)
        torch.testing.assert_close(actual[:, 2], values[(i+1) % 4], rtol=0, atol=0)
    cache.close()
    clone.close()


def test_missing_invalid_and_incomplete_cache_fail(tmp_path):
    cache_files(tmp_path, {0: torch.zeros(2, 2, 4, dtype=torch.bfloat16),
                           2: torch.zeros(2, 2, 4, dtype=torch.bfloat16)})
    cache = ImageFrameCache(tmp_path)
    with pytest.raises(ValueError, match="Incomplete"):
        cache.require_complete()
    with pytest.raises(Exception, match="1"):
        cache.read_many([1])
    with pytest.raises(ValueError, match="identity"):
        ImageFrameCache(tmp_path, identity={})


def test_episode_start_history_and_episode_tail_padding():
    reader = SimpleNamespace(_cum_n_starts=np.array([0, 40, 45]),
                             _eps_df=pd.DataFrame({"length": [40, 5]}))
    assert window_frame_indices(reader, 0, 0) == [0, 0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert window_frame_indices(reader, 0, 39) == [35, *([39] * 9)]
    assert window_frame_indices(reader, 1, 0) == [40, 40, *([44] * 8)]


class SpatialIJEPA(_MockIJEPA):
    def forward(self, x):
        tokens = super().forward(x)
        pixels = x[:, 0, ::self.patch, ::self.patch].flatten(1).unsqueeze(-1)
        return tokens + pixels * torch.arange(self.embed_dim, device=x.device, dtype=x.dtype)


@pytest.mark.parametrize("experiment", ["E01", "E02"])
def test_cached_and_online_loss_gradients_and_strict_reload(tmp_path, monkeypatch, experiment):
    cfg, profile, _ = resolve_experiment(experiment)
    vb = cfg.model.video_backbone
    vb.text_encoder_type = "umt5-xxl"
    vb.text_embedding_dir = str(make_cache(tmp_path))
    if experiment == "E01":
        source = tmp_path / "dino"
        DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_register_tokens=4, patch_size=16)).save_pretrained(source)
        vb.encoder.model_path = str(source)
    else:
        monkeypatch.setattr(IJEPAImageVideoEncoder, "from_pretrained", classmethod(
            lambda cls, path, **kw: cls(SpatialIJEPA(), embed_dim=8, **{
                k: v for k, v in kw.items() if k not in ("checkpoint", "model_name", "img_size", "patch_size")})))
    for key, value in dict(dim=96, ffn_dim=192, num_heads=4, num_layers=1, freq_dim=32).items():
        vb.dit_config[key] = value
    cfg.model.action_backbone.dim = 32
    cfg.model.action_backbone.ffn_dim = 64
    cfg.dataloader.normalize_mode = None
    spec = resolve_architecture_config(cfg.model)
    model = build_architecture(spec.registry_name, spec.params)
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    model.train()
    model.freeze_modules(list(cfg.model.freeze))
    model.init_training_schedulers()
    encoder = model.video_backbone.video_encoder
    if experiment == "E02":
        with torch.no_grad():
            encoder.delta_scale.fill_(0.2)
    clip = frames()
    sample = dict(video=clip[1:], clean_condition_images=clip[:2], prompt="open drawer",
                  action=torch.zeros(32, 80), proprio=torch.zeros(80), video_mask=torch.ones(9, dtype=torch.bool))
    online = model.prepare_inputs([copy.deepcopy(sample)])
    with torch.no_grad():
        raw = encoder._encode_frames(encoder.preprocess_video(clip))[0]
    cache_files(tmp_path / "cache", {i: raw[:, i].contiguous() for i in range(10)})
    cached_sample = {**sample, "video": [], "clean_condition_images": None, "image_frame_indices": list(range(10))}
    cached_sample = ImageFrameCacheCollator(tmp_path / "cache")([cached_sample])[0]
    # A cache hit must not invoke the image backbone at all.
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached training ran the frozen backbone")
    monkeypatch.setattr(encoder._m, "forward", forbidden)
    cached = model.prepare_inputs([cached_sample])
    assert cached.keys() == online.keys()
    for key, expected in online.items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(cached[key], expected, rtol=0, atol=0)
    gradients, losses = [], []
    for inputs in (online, cached):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(77)
        result = model.compute_loss(**copy.deepcopy(inputs))
        result["loss"].backward()
        losses.append(result["loss"].detach())
        gradients.append({name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None})
        assert all(p.grad is None for p in encoder._m.parameters())
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
    projection = model.video_backbone.dit.patch_embedding
    assert projection.proj1.weight.grad.abs().sum() > 0
    if experiment == "E02":
        for p in (encoder.delta_scale, encoder._delta_proj.weight, encoder.feature_norm.weight):
            assert p.grad is not None and p.grad.abs().sum() > 0
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(saved, strict=True)
    assert encoder.properties.num_condition_latents == profile["geometry"]["clean_latents"]


@pytest.mark.parametrize("name", ["E01", "E02"])
def test_cache_profiles_keep_original_model_and_training_budget(name):
    original, before, _ = resolve_experiment(name)
    cached, after, _ = resolve_experiment(name + "-Cache")
    assert OmegaConf.to_container(original.model) == OmegaConf.to_container(cached.model)
    assert before["geometry"] == after["geometry"]
    for key in ("batch_size", "num_epochs", "learning_rate", "lambda_video", "lambda_action",
                "gradient_accumulation_steps", "zero_stage", "use_gradient_checkpointing"):
        assert original.training[key] == cached.training[key]
    assert cached.dataloader.frame_feature_cache_dir
