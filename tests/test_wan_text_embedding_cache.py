import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.model.video_backbone.wan import component_specs, loader
from openwam.model.video_backbone.wan.encode import encode_text
from openwam.model.video_backbone.wan.text_embedding_cache import TextEmbeddingCache


def make_cache(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    files = {}
    for name, value in (("open drawer", 1), ("close drawer", 2)):
        context = torch.zeros(512, 4096, dtype=torch.bfloat16)
        context[:3] = value
        files[name] = f"{value}.safetensors"
        save_file({"context": context, "seq_len": torch.tensor(3)}, str(directory / files[name]))
    (directory / "manifest.json").write_text(json.dumps({"version": 1, "shape": [512, 4096], "prompts": files}))
    return directory


class TestWanTextEmbeddingCache(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.cache_dir = make_cache(self.tmp_path)

    def test_cache_preserves_batch_order_repetition_and_zero_padding(self):
        cache_dir = self.cache_dir
        cache = TextEmbeddingCache(cache_dir)
        context, lengths = encode_text(
            ["close drawer", "open drawer", "close drawer"],
            tokenizer=None, text_encoder=None, device="cpu", embedding_cache=cache,
        )
        assert context.dtype == torch.bfloat16
        assert context.shape == (3, 512, 4096)
        assert context[:, 0, 0].tolist() == [2, 1, 2]
        assert lengths.tolist() == [3, 3, 3]
        assert torch.count_nonzero(context[:, 3:]) == 0
        with self.assertRaisesRegex(KeyError, "no prompt"):
            cache.encode(["unknown instruction"], device="cpu")

    def test_online_text_path_remains_available(self):
        def tokenizer(prompts, **kwargs):
            return torch.tensor([[4, 5, 0]]), torch.tensor([[1, 1, 0]])

        context, lengths = encode_text(
            ["example"], tokenizer=tokenizer,
            text_encoder=lambda ids, mask: ids.float().unsqueeze(-1), device="cpu",
        )
        assert context.squeeze(-1).tolist() == [[4, 5, 0]]
        assert lengths.tolist() == [2]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cached_features_move_between_devices_without_changing_values(self):
        cache = TextEmbeddingCache(self.cache_dir)
        prompts = ["close drawer", "open drawer", "close drawer"]
        expected, expected_lengths = cache.encode(prompts, device="cpu")
        with patch("openwam.model.video_backbone.wan.text_embedding_cache.load_file", side_effect=AssertionError("cache reread")):
            for device in ("cuda", "cuda", "cpu"):
                actual, lengths = cache.encode(prompts, device=device)
                assert actual.device.type == torch.device(device).type
                assert torch.equal(actual.cpu(), expected)
                assert torch.equal(lengths.cpu(), expected_lengths)

    def test_training_dict_with_dictconfig_backbone_skips_t5(self):
        source = {"video_backbone": OmegaConf.create({
            "model_path": "/unused/pretrained", "text_embedding_dir": str(self.cache_dir),
        })}
        with patch.object(loader, "build_holder_from_model_path", return_value=SimpleNamespace()) as build:
            holder = loader.build_holder(source)
        assert build.call_args.kwargs["skip_text_encoder"] is True
        assert len(holder.text_embedding_cache.files) == 2


    def test_checkpoint_references_shared_cache_without_copying_or_constructing_t5(self):
        cache_dir, tmp_path = self.cache_dir, self.tmp_path
        cfg = OmegaConf.create({"model": {"video_backbone": {
            "model_path": str(tmp_path), "text_embedding_dir": str(cache_dir),
        }}})
        specs = {"components": [{"attr": "text_encoder", "model_class": "must.never.import.T5"}],
                 "tokenizer": {"class": "must.never.import.Tokenizer"}}
        checkpoint = tmp_path / "checkpoint"
        with patch.object(component_specs, "generate_video_backbone_component_specs", return_value=specs):
            component_specs.save_video_backbone_deploy_assets(str(checkpoint), cfg)
        assert not (checkpoint / "text_embedding").exists()
        assert cfg.model.video_backbone.text_embedding_dir == str(cache_dir.resolve())
        assert cfg.model.video_backbone.components == []
        assert "tokenizer" not in cfg.model.video_backbone
        source = OmegaConf.to_container(cfg.model.video_backbone)
        # A stale checkpoint-local cache must not override the configured source.
        stale_cache = checkpoint / "text_embedding"
        stale_cache.mkdir(parents=True)
        (stale_cache / "manifest.json").write_text("invalid stale cache")
        holder = loader.build_holder(source, ckpt_dir=str(checkpoint))
        assert holder.text_encoder is None and holder.tokenizer is None
        context, lengths = holder.text_embedding_cache.encode(["open drawer"], device="cpu")
        assert context[0, 0, 0].item() == 1 and lengths.item() == 3


if __name__ == "__main__":
    unittest.main()
