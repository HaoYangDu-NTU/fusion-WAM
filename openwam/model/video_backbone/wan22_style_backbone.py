"""Scratch-only Wan2.2 capacity ablation; all runtime behavior stays in Wan22Ti2v."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

logger = logging.getLogger(__name__)


class Wan22Style(Wan22Ti2v):
    """Build a smaller Wan DiT directly, without loading a 5B checkpoint first."""

    @classmethod
    def from_pretrained(cls, source, *, external_encoder=None, text_dim=None, **kw):
        # Support both full Hydra configs and resolve_architecture_config's
        # normalized model dict, as well as saved deploy component configs.
        model_cfg = source.get("model", source)
        vb_cfg = model_cfg.get("video_backbone", model_cfg)
        vb_cfg = OmegaConf.to_container(vb_cfg, resolve=True) if isinstance(vb_cfg, DictConfig) else dict(vb_cfg)
        if not vb_cfg.get("from_scratch"):
            raise ValueError("wan22_style requires from_scratch=true; no pretrained DiT weights are used")
        if external_encoder is None:
            raise ValueError("wan22_style requires a configured external video encoder")
        from openwam.text_condition.config import cache_directory, text_spec

        spec_text = text_spec(vb_cfg.get("text_encoder_type", "umt5-xxl"),
                              vb_cfg.get("vlm_text_embedding_method", "avg_all_layers"))
        if not vb_cfg.get("text_embedding_dir"):
            vb_cfg["text_embedding_dir"] = str(cache_directory(
                vb_cfg.get("text_encoder_type", "umt5-xxl"),
                vb_cfg.get("vlm_text_embedding_method", "avg_all_layers"),
                data_root=Path(__file__).resolve().parents[3] / "data"))

        dit_config = dict(vb_cfg["dit_config"])
        dit_config["text_dim"] = spec_text["shape"][1]
        vb_cfg["dit_config"] = dict(dit_config)
        spec = external_encoder.properties
        dit_config.update(in_dim=spec.z_dim, out_dim=spec.z_dim, patch_size=list(spec.dit_patch_size))
        dit_entry = {
            "attr": "dit",
            "model_class": "openwam.model.video_backbone.wan.models.dit.WanModel",
            "extra_kwargs": dit_config,
        }
        if "components" in vb_cfg:
            saved_dit = next((entry for entry in vb_cfg["components"] if entry.get("attr") == "dit"), None)
            if saved_dit != dit_entry:
                raise ValueError("Saved DiT component does not match wan22_style dit_config and encoder geometry")
        else:
            vb_cfg["components"] = [dit_entry]

        # The normal Wan component loader constructs WanModel directly from
        # its kwargs, uses the shared text cache, and skips the native VAE.
        # Architecture initialization still performs the existing encoder I/O
        # adaptation and official from-scratch parameter reset afterwards.
        backbone = super().from_pretrained(
            {"video_backbone": vb_cfg}, external_encoder=external_encoder, text_dim=text_dim, **kw
        )
        backbone._scratch_dit_entry = dit_entry
        return backbone

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        # Scanning the old model_path would describe a 5B DiT. Persist the
        # actual construction spec instead. The external-encoder marker is
        # required by BaseWAMArchitecture._build_external_encoder_skeleton;
        # V-JEPA itself rebuilds from its checkpoint-local manifest.
        encoder_cls = type(self.video_encoder)
        components = [self._scratch_dit_entry, {
            "attr": "vae",
            "model_class": f"{encoder_cls.__module__}.{encoder_cls.__name__}",
            "source": "state_dict",
            "extra_kwargs": {},
        }]
        from openwam.text_condition.config import save_text_cache_config

        save_text_cache_config(cfg, self._text_embedding_cache)
        cache_dir = os.path.abspath(self._text_embedding_cache.directory)
        with open_dict(cfg):
            cfg.model.video_backbone.components = components
            cfg.model.video_backbone.dit_config.text_dim = self._scratch_dit_entry["extra_kwargs"]["text_dim"]
            cfg.model.video_backbone.text_embedding_dir = cache_dir
            if "tokenizer" in cfg.model.video_backbone:
                del cfg.model.video_backbone.tokenizer
        # Match the shared-cache contract of the other Wan backbones. Copying
        # the tree also follows links to unrelated, potentially huge caches.
        logger.info("Referencing shared text embeddings: %s", cache_dir)
        self.video_encoder.save_deploy_assets(output_dir, cfg)
