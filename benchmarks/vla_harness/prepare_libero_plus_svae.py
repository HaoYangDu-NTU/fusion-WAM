"""Prepare a Plus deployment preserving native visual-encoder assets.

Reads checkpoint headers only. Does not load model weights or encode text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from safetensors import safe_open

from benchmarks.libero_plus_instructions import parse_bool
from benchmarks.vla_harness.checkpoint_contract import (
    validate_feature_fusion_checkpoint, validate_dinov3_checkpoint, validate_siglip_checkpoint, validate_ijepa_checkpoint, validate_wan21_features_checkpoint, validate_wan_vae_checkpoint, validate_flux2_framewise_checkpoint, validate_vggt_omega_checkpoint,
)

ENCODER_MARKER = {
    "attr": "vae",
    "model_class": "openwam.model.video_backbone.encoder.vjepa21.VJEPA21VideoEncoder",
    "source": "state_dict",
    "extra_kwargs": {},
}


def prepare_checkpoint(source_dir, checkpoint_name, output_dir, text_embedding_dir=None, *, corrected=True):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("Use a separate deployment directory and a checkpoint filename")
    cfg = yaml.safe_load((source_dir / "config.yaml").read_text())
    vb = cfg["model"]["video_backbone"]
    encoder_name = vb.get("encoder", {}).get("name")
    native_assets = {
        "vjepa21": ["manifest.json"], "dinov3": ["dinov3/config.json", "dinov3/encoder.json"],
        "ijepa": ["ijepa/encoder.json"],
        "wan21_features": ["wan21_features/encoder.json"],
        "wan_vae": ["wan_vae/encoder.json"],
        "vggt_omega": ["vggt_omega/encoder.json"],
        "flux2_framewise": ["flux2_framewise/config.json", "flux2_framewise/encoder.json"],
        "siglip": ["siglip/config.json", "siglip/encoder.json"],
    }
    if encoder_name not in native_assets:
        raise ValueError(f"Add the encoder's native Plus preparation for {encoder_name}")
    assets = [*native_assets[encoder_name], "normalization_stats.npy", checkpoint_name]
    for name in ["config.yaml", *assets]:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(source_dir / name)
    cfg = yaml.safe_load((source_dir / "config.yaml").read_text())
    vb = cfg["model"]["video_backbone"]
    from openwam.text_condition.config import cache_directory, text_spec

    kind = vb.get("text_encoder_type", "umt5-xxl")
    method = vb.get("vlm_text_embedding_method", "avg_all_layers")
    # A prepared Plus deployment may be passed through the launcher again.
    # Reuse only a cache explicitly marked for the requested instruction mode;
    # a standard LIBERO cache must still resolve to the Plus default.
    configured = Path(vb.get("text_embedding_dir") or "/nonexistent")
    configured_manifest = configured / "manifest.json"
    if (text_embedding_dir is None and (source_dir / "deployment_preparation.json").is_file()
            and configured_manifest.is_file()):
        metadata = json.loads(configured_manifest.read_text())
        if metadata.get("correct_instruction_suffix") is corrected:
            text_embedding_dir = configured
    text_embedding_dir = Path(text_embedding_dir or cache_directory(
        kind, method, "libero-plus", data_root=Path(__file__).resolve().parents[2] / "data",
        corrected=corrected)).resolve()
    cache_manifest = text_embedding_dir / "manifest.json"
    if cache_manifest.is_file():
        text_manifest = json.loads(cache_manifest.read_text())
        fingerprint = vb.get("text_embedding_fingerprint")
        if fingerprint is not None and text_manifest.get("model_fingerprint") != fingerprint:
            raise ValueError("Plus text cache model fingerprint differs from training")
        if text_manifest.get("version") == 2:
            for key, value in text_spec(kind, method).items():
                if text_manifest.get(key) != value:
                    raise ValueError(f"Plus text cache {key} differs from training config")
        elif kind != "umt5-xxl":
            raise ValueError("A VLM checkpoint cannot use a legacy UMT5 cache")
    if not vb.get("from_scratch"):
        raise ValueError("Expected an external-encoder checkpoint trained from scratch")
    if not vb.get("text_embedding_dir"):
        raise ValueError("Expected the cached-text training/deployment configuration")
    fusion = validate_feature_fusion_checkpoint(source_dir, checkpoint_name, cfg)
    visual = (validate_wan_vae_checkpoint(source_dir, checkpoint_name, cfg)
              or validate_wan21_features_checkpoint(source_dir, checkpoint_name, cfg)
              or validate_dinov3_checkpoint(source_dir, checkpoint_name, cfg)
              or validate_ijepa_checkpoint(source_dir, checkpoint_name, cfg)
              or validate_flux2_framewise_checkpoint(source_dir, checkpoint_name, cfg)
              or validate_vggt_omega_checkpoint(source_dir, checkpoint_name, cfg)
              or validate_siglip_checkpoint(source_dir, checkpoint_name, cfg))
    prefix = "video_backbone.video_encoder."
    with safe_open(source_dir / checkpoint_name, framework="np") as checkpoint:
        keys = set(checkpoint.keys())
        has_svae = (
            any(key.startswith(prefix + "_svae.") for key in keys)
            or (source_dir / "svae_config.json").exists()
            or vb["encoder"].get("svae_path") is not None
            or vb["encoder"].get("svae_target_dim") is not None
        )
        input_dim = visual["latent_channels"] if visual else int(
            json.loads((source_dir / "manifest.json").read_text())["embed_dim"]
        )
        latent_dim = fusion["latent_channels"] if fusion else input_dim
        expected = {}
        if has_svae:
            svae = json.loads((source_dir / "svae_config.json").read_text())["model_config"]
            latent_dim, input_dim = int(svae["latent_dim"]), int(svae["input_dim"])
            target = vb["encoder"].get("svae_target_dim")
            if target is not None and int(target) != latent_dim:
                raise ValueError("S-VAE sidecar and encoder.svae_target_dim disagree")
            assets.append("svae_config.json")
            expected.update({
                prefix + "_svae.enc_proj.weight": [2 * latent_dim, input_dim],
                prefix + "_svae.input_mean": [input_dim],
                prefix + "_svae.input_std": [input_dim],
            })
        if encoder_name not in ("dinov3", "siglip", "wan21_features", "wan_vae", "vggt_omega", "flux2_framewise"):
            expected[prefix + "feature_norm.weight"] = [input_dim if fusion else latent_dim]
        patch_key = "video_backbone.dit.patch_embedding.weight"
        if patch_key not in keys and (vb["encoder"].get("add_input_bottleneck", False) or encoder_name == "siglip"):
            patch_key = "video_backbone.dit.patch_embedding.clean_proj.weight"
            if patch_key not in keys:
                patch_key = "video_backbone.dit.patch_embedding.proj1.weight"
        if patch_key not in keys:
            raise ValueError("Checkpoint is missing DiT patch embedding weights")
        patch_shape = checkpoint.get_slice(patch_key).get_shape()
        if patch_shape[1] != latent_dim:
            raise ValueError("DiT input channels do not match the encoder latent dimension")
        if vb["encoder"].get("seperate_encode_view", False):
            hidden_dim = patch_shape[0]
            if patch_key.endswith("proj1.weight"):
                hidden_dim = checkpoint.get_slice("video_backbone.dit.patch_embedding.proj2.weight").get_shape()[0]
            expected["video_backbone.dit.view_embedding.weight"] = [2, hidden_dim]
        for key, shape in expected.items():
            if key not in keys or checkpoint.get_slice(key).get_shape() != shape:
                raise ValueError(f"Encoder checkpoint/config mismatch: {key}, expected {shape}")

    # Same structural marker as the completed 96.00% LIBERO S-VAE run.
    # from_skeleton reads manifest.json / svae_config.json; it does not load a
    # native Wan VAE or the training-time svae.pt from this marker.
    components = vb["components"]
    if not any(entry.get("attr") == "vae" for entry in components):
        if encoder_name != "vjepa21":
            raise ValueError("Missing native encoder reconstruction component")
        components.append(dict(ENCODER_MARKER))
    # Set both paths so a not-yet-encoded Plus cache cannot fall back to LIBERO.
    vb["text_embedding_dir"] = str(text_embedding_dir)
    links = {name: source_dir / name for name in assets}
    links["text_embedding"] = text_embedding_dir
    if output_dir.exists():
        if yaml.safe_load((output_dir / "config.yaml").read_text()) != cfg:
            raise ValueError(f"Refusing to overwrite a different deployment config: {output_dir}")
        for name, target_path in links.items():
            if not (output_dir / name).is_symlink() or (output_dir / name).resolve() != target_path.resolve():
                raise ValueError(f"Existing deployment asset differs: {output_dir / name}")
        return output_dir

    output_dir.mkdir(parents=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name, target_path in links.items():
        (output_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / name).symlink_to(target_path)
    record = {
        "source_dir": str(source_dir),
        "checkpoint_name": checkpoint_name,
        "text_embedding_dir": str(text_embedding_dir),
        "svae_input_dim": input_dim if has_svae else None,
        "svae_latent_dim": latent_dim if has_svae else None,
        "encoder_latent_dim": latent_dim,
        "checkpoint_tensor_count": len(keys),
        "weights_loaded": False,
        "text_encoding_performed": False,
    }
    (output_dir / "deployment_preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-embedding-dir", type=Path)
    parser.add_argument("--correct-instruction-suffix", type=parse_bool, default=True)
    args = parser.parse_args()
    print(prepare_checkpoint(args.source_dir, args.ckpt_name, args.output_dir, args.text_embedding_dir,
                             corrected=args.correct_instruction_suffix))


if __name__ == "__main__":
    main()
