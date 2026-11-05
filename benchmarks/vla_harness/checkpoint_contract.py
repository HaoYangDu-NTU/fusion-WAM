"""Header-only checks for expanded or learned V-JEPA features in either benchmark."""

import json
import math
from pathlib import Path

from safetensors import safe_open


def validate_flux2_framewise_checkpoint(source_dir, checkpoint_name, cfg):
    encoder = cfg.get("model", {}).get("video_backbone", {}).get("encoder") or {}
    if encoder.get("name") != "flux2_framewise":
        return None
    from openwam.model.video_backbone.encoder.flux2_framewise import Flux2FramewiseEncoder
    skeleton = Flux2FramewiseEncoder.from_skeleton({}, device="meta", encoder_cfg=encoder, ckpt_dir=str(source_dir))
    with safe_open(Path(source_dir) / checkpoint_name, framework="np") as weights:
        prefix = "video_backbone.video_encoder."
        for name, tensor in skeleton.state_dict().items():
            if weights.get_slice(prefix + name).get_shape() != list(tensor.shape):
                raise ValueError(f"FLUX.2 encoder shape mismatch: {name}")
        if any("decoder." in key for key in weights.keys() if key.startswith(prefix)):
            raise ValueError("E06 checkpoint unexpectedly includes a decoder")
        patch = weights.get_slice("video_backbone.dit.patch_embedding.clean_proj.weight").get_shape()
        if patch[1:] != [128, 1, 2, 2]:
            raise ValueError("E06 projection shape mismatch")
        if weights.get_slice("video_backbone.dit.head.head.weight").get_shape() != [512, patch[0]]:
            raise ValueError("E06 output shape mismatch")
    return dict(name="flux2_framewise", latent_channels=128, patch_size=[1, 2, 2],
                temporal_compression=1, num_condition_latents=2)


def validate_vggt_omega_checkpoint(source_dir, checkpoint_name, cfg):
    encoder = cfg.get("model", {}).get("video_backbone", {}).get("encoder") or {}
    if encoder.get("name") != "vggt_omega":
        return None
    saved = json.loads((Path(source_dir) / "vggt_omega/encoder.json").read_text())
    channels = saved["channels"]
    processing = saved["processing"]
    if (processing.get("cross_time_encoding") is not False
            or processing.get("joint_views_per_timestamp") is not True
            or processing.get("output") != "final_spatial_patches_only"
            or processing.get("add_one_history_context") is not True
            or encoder.get("add_one_history_context", False) is not True
            or processing.get("view_order") != ["front", "wrist"]
            or processing.get("output_layout") != "width_concat"):
        raise ValueError("Expected frozen spatial-only per-time stereo E05")
    for key, default in (("pair_microbatch_size", 1), ("bottleneck_dim", 256), ("feature_norm_eps", 1e-5), ("image_size", 224)):
        if processing[key] != encoder.get(key, default):
            raise ValueError(f"E05 checkpoint/config differs: {key}")
    with safe_open(Path(source_dir) / checkpoint_name, framework="np") as weights:
        keys = set(weights.keys())
        prefix = "video_backbone.video_encoder._m."
        for name, shape in (("camera_token", [1, 2, 1, channels // 2]),
                            ("register_token", [1, 2, 16, channels // 2])):
            if prefix + name not in keys or weights.get_slice(prefix + name).get_shape() != shape:
                raise ValueError(f"Native E05 token structure differs: {name}")
        if not any(k.startswith(prefix + "frame_blocks.") for k in keys):
            raise ValueError("Missing native E05 frame blocks")
        if not any(k.startswith(prefix + "inter_frame_blocks.") for k in keys):
            raise ValueError("Missing native E05 inter-frame blocks")
        if any(k.startswith(prefix + name) for k in keys for name in ("camera_head.", "dense_head.", "text_alignment_head.")):
            raise ValueError("E05 encoder unexpectedly contains task heads")
        patch = weights.get_slice("video_backbone.dit.patch_embedding.clean_proj.weight").get_shape()
        if patch[1:] != [channels, 1, 2, 2]:
            raise ValueError("E05 input projection shape differs")
        if weights.get_slice("video_backbone.dit.head.head.weight").get_shape() != [channels * 4, patch[0]]:
            raise ValueError("E05 output projection shape differs")
    return dict(name="vggt_omega", latent_channels=channels, patch_size=[1, 2, 2],
                temporal_compression=1, num_condition_latents=2)


def validate_wan_vae_checkpoint(source_dir, checkpoint_name, cfg):
    encoder = cfg.get("model", {}).get("video_backbone", {}).get("encoder") or {}
    if encoder.get("name") != "wan_vae":
        return None
    saved = json.loads((Path(source_dir) / "wan_vae/encoder.json").read_text())
    processing = saved["processing"]
    for key in ("vae_version", "microbatch_size", "bottleneck_dim"):
        if encoder.get(key) != processing[key]:
            raise ValueError(f"Wan VAE checkpoint/config differs: {key}")
    if processing.get("encoder_only") is not True or processing.get("normalization") != "native_vae_mean_std":
        raise ValueError("Expected native encoder-only VAE contract")
    version = processing["vae_version"]
    expected_geometry = {"2.1": (16, 8), "2.2": (48, 16)}
    if version not in expected_geometry or (saved["z_dim"], saved["spatial_compression"]) != expected_geometry[version]:
        raise ValueError("Wan VAE version/geometry mismatch")
    channels = saved["z_dim"]
    spatial_patch = processing.get("spatial_patch_size", 2)
    if spatial_patch != encoder.get("spatial_patch_size", 2) or spatial_patch not in (2, 4):
        raise ValueError("Wan VAE spatial patch/config mismatch")
    area = spatial_patch ** 2
    with safe_open(Path(source_dir) / checkpoint_name, framework="np") as weights:
        keys = set(weights.keys())
        prefix = "video_backbone.video_encoder._m.model."
        if not any(k.startswith(prefix + "encoder.") for k in keys) or not any(k.startswith(prefix + "conv1.") for k in keys):
            raise ValueError("Missing Wan VAE encoder weights")
        if any(k.startswith(prefix + "decoder.") or k.startswith(prefix + "conv2.") for k in keys):
            raise ValueError("VAE encoder checkpoint unexpectedly includes decoder weights")
        patch = weights.get_slice("video_backbone.dit.patch_embedding.clean_proj.weight").get_shape()
        if patch[1:] != [channels, 1, spatial_patch, spatial_patch]:
            raise ValueError("Wan VAE input projection channel/patch mismatch")
        if weights.get_slice("video_backbone.dit.head.head.weight").get_shape() != [channels * area, patch[0]]:
            raise ValueError("Wan VAE output projection mismatch")
    return dict(name="wan_vae", latent_channels=channels, patch_size=[1, spatial_patch, spatial_patch],
                temporal_compression=4, num_condition_latents=1)


def validate_wan21_features_checkpoint(source_dir, checkpoint_name, cfg):
    encoder = cfg.get("model", {}).get("video_backbone", {}).get("encoder") or {}
    if encoder.get("name") != "wan21_features":
        return None
    saved = json.loads((Path(source_dir) / "wan21_features/encoder.json").read_text())
    processing = saved["processing"]
    for key in ("feature_layer", "microbatch_size", "bottleneck_dim", "feature_norm_eps"):
        if encoder.get(key) != processing[key]:
            raise ValueError(f"Wan video feature checkpoint/config differs: {key}")
    if (processing.get("timestep") != 0 or processing.get("temporal_design") != "native_video_independent_clean"
            or processing.get("clean_condition_frame_offsets") != [-4, -3, -2, -1, 0]):
        raise ValueError("Invalid Wan video feature temporal/extraction contract")
    channels = saved["model_config"]["dim"]
    prefix = "video_backbone.video_encoder."
    with safe_open(Path(source_dir) / checkpoint_name, framework="np") as weights:
        expected = {prefix + "dit.patch_embedding.weight": [channels, 36, 1, 2, 2],
                    prefix + "empty_context": [1, 512, 4096]}
        for key, shape in expected.items():
            if key not in weights.keys() or weights.get_slice(key).get_shape() != shape:
                raise ValueError(f"Wan video feature checkpoint missing/mismatched: {key}")
        head = weights.get_slice("video_backbone.dit.head.head.weight").get_shape()
        hidden = head[1]
        projections = {"head.head.weight": [channels * 4, hidden],
                       "patch_embedding.clean_proj.weight": [hidden, channels, 1, 2, 2],
                       "patch_embedding.proj1.weight": [processing["bottleneck_dim"], channels, 1, 2, 2],
                       "patch_embedding.proj2.weight": [hidden, processing["bottleneck_dim"], 1, 1, 1],
                       "view_embedding.weight": [2, hidden]}
        for key, shape in projections.items():
            full_key = "video_backbone.dit." + key
            if full_key not in weights.keys() or weights.get_slice(full_key).get_shape() != shape:
                raise ValueError(f"Wan video feature policy projection mismatch: {key}")
    return dict(name="wan21_features", latent_channels=channels, patch_size=[1, 2, 2],
                temporal_compression=4, num_condition_latents=1)


def validate_ijepa_checkpoint(source_dir, checkpoint_name, cfg):
    """IJEPA owns its affine normalization and learned history difference."""
    vb = cfg.get("model", {}).get("video_backbone", {})
    encoder = vb.get("encoder") or {}
    if encoder.get("name") != "ijepa":
        return None
    saved = json.loads((Path(source_dir) / "ijepa/encoder.json").read_text())
    settings = {key: encoder.get(key, default) for key, default in {
        "patch_size": 16, "temporal_group_size": 2, "history_fusion_mode": "gated_difference",
        "add_one_history_context": True, "seperate_encode_view": False,
        "add_input_bottleneck": True, "bottleneck_dim": 256, "feature_norm_eps": 1e-5,
    }.items()}
    if saved["processing"] != settings:
        raise ValueError("I-JEPA processing config differs from saved encoder.json")
    channels = saved["embed_dim"]
    prefix = "video_backbone.video_encoder."
    dit = "video_backbone.dit."
    with safe_open(Path(source_dir) / checkpoint_name, framework="np") as weights:
        hidden = weights.get_slice(dit + "head.head.weight").get_shape()[1]
        expected = {
            prefix + "feature_norm.weight": [channels], prefix + "feature_norm.bias": [channels],
            prefix + "_delta_proj.weight": [channels, channels], prefix + "delta_scale": [],
            dit + "head.head.weight": [4 * channels, hidden],
        }
        if settings["add_input_bottleneck"]:
            expected.update({
                dit + "patch_embedding.clean_proj.weight": [hidden, channels, 1, 2, 2],
                dit + "patch_embedding.proj1.weight": [settings["bottleneck_dim"], channels, 1, 2, 2],
                dit + "patch_embedding.proj2.weight": [hidden, settings["bottleneck_dim"], 1, 1, 1],
            })
        if settings["seperate_encode_view"]:
            expected[dit + "view_embedding.weight"] = [2, hidden]
        for key, shape in expected.items():
            if key not in weights.keys() or weights.get_slice(key).get_shape() != shape:
                raise ValueError(f"I-JEPA checkpoint/config mismatch: {key}, expected {shape}")
    return dict(name="ijepa", latent_channels=channels, patch_size=[1, 2, 2],
                temporal_compression=settings["temporal_group_size"],
                num_condition_latents=2 if settings["history_fusion_mode"] == "two_clean" else 1)


def validate_feature_fusion_checkpoint(source_dir, checkpoint_name, cfg):
    """Return a checked contract for concat/AdaFuse, None for legacy static paths."""
    source_dir = Path(source_dir)
    vb = cfg.get("model", {}).get("video_backbone", {})
    encoder = vb.get("encoder") or {}
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    fusion = manifest.get("feature_fusion", {})
    method = "concat_before_bottleneck"
    method_keys = ("condition_fusion_method", "future_fusion_method")
    if any(source.get(key) == "adafuse" for source in (encoder, fusion) for key in method_keys):
        return _validate_adafuse_checkpoint(source_dir, checkpoint_name, cfg, manifest)
    if not any(source.get(key) == method for source in (encoder, fusion) for key in method_keys):
        return None

    from openwam.model.video_backbone.encoder.vjepa21_fusion import concat_feature_metadata, resolve_fusion_layers

    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("Expected a checkpoint filename")
    if (encoder.get("name") != "vjepa21" or not vb.get("from_scratch")
            or not encoder.get("add_input_bottleneck")
            or encoder.get("svae_path") is not None or encoder.get("svae_target_dim") is not None
            or (source_dir / "svae_config.json").exists()
            or cfg["model"].get("architecture", {}).get("repa_enabled", False)):
        raise ValueError("Concat requires input bottleneck, raw V-JEPA and REPA disabled")
    depth = fusion.get("encoder_depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
        raise ValueError("Missing concat encoder_depth in checkpoint manifest")
    input_dim = int(manifest["embed_dim"])
    layers = resolve_fusion_layers(encoder.get("fusion_layers", ()), depth)
    expected_fusion = {
        **dict.fromkeys(method_keys, method),
        "fusion_layers": list(layers),
        **concat_feature_metadata(layers, input_dim, depth),
    }
    if (layers[-1] != depth - 1 or fusion != expected_fusion
            or any(encoder.get(key) != method for key in method_keys)):
        raise ValueError("Concat feature fusion config differs from the checkpoint manifest")

    latent_dim = fusion["latent_channels"]
    patch = (1, 2, 2) if encoder.get("use_wan_style_patch_grid", True) else (1, 1, 1)
    prefix = "video_backbone.video_encoder."
    dit = "video_backbone.dit."
    with safe_open(source_dir / checkpoint_name, framework="np") as checkpoint:
        keys = set(checkpoint.keys())
        if any(key.startswith(prefix + "_svae.") for key in keys):
            raise ValueError("Concat requires raw V-JEPA without S-VAE weights")
        if any(dit + "patch_embedding." + name + ".weight" not in keys
               for name in ("clean_proj", "proj1", "proj2")):
            raise ValueError("Concat checkpoint is missing clean/future projection weights")
        hidden = checkpoint.get_slice(dit + "patch_embedding.proj2.weight").get_shape()[0]
        if hidden != (vb.get("dit_config") or {}).get("dim", hidden):
            raise ValueError("DiT hidden dimension differs from the saved config")
        bottleneck = encoder.get("bottleneck_dim", 512)
        expected = {
            prefix + "feature_norm.weight": [input_dim],
            prefix + "feature_norm.bias": [input_dim],
            dit + "patch_embedding.clean_proj.weight": [hidden, latent_dim, *patch],
            dit + "patch_embedding.proj1.weight": [bottleneck, latent_dim, *patch],
            dit + "patch_embedding.proj2.weight": [hidden, bottleneck, 1, 1, 1],
            dit + "head.head.weight": [latent_dim * math.prod(patch), hidden],
            dit + "head.head.bias": [latent_dim * math.prod(patch)],
        }
        if encoder.get("seperate_encode_view", False):
            expected[dit + "view_embedding.weight"] = [2, hidden]
        for key, shape in expected.items():
            if key not in keys or checkpoint.get_slice(key).get_shape() != shape:
                raise ValueError(f"Encoder checkpoint/config mismatch: {key}, expected {shape}")
    return fusion


def _validate_adafuse_checkpoint(source_dir, checkpoint_name, cfg, manifest):
    from openwam.model.video_backbone.encoder.vjepa21_fusion import (
        FUSION_METHODS, adafuse_feature_metadata, resolve_fusion_layers,
    )

    vb = cfg["model"]["video_backbone"]
    encoder, fusion = vb["encoder"], manifest.get("feature_fusion", {})
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("Expected a checkpoint filename")
    if (encoder.get("name") != "vjepa21" or not vb.get("from_scratch")
            or encoder.get("svae_path") is not None or encoder.get("svae_target_dim") is not None
            or (source_dir / "svae_config.json").exists()
            or cfg["model"].get("architecture", {}).get("repa_enabled", False)):
        raise ValueError("AdaFuse requires raw V-JEPA and REPA disabled")
    depth = fusion.get("encoder_depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
        raise ValueError("Missing AdaFuse encoder_depth in checkpoint manifest")
    dim = int(manifest["embed_dim"])
    layers = resolve_fusion_layers(encoder.get("fusion_layers", ()), depth)
    methods = {f"{branch}_fusion_method": encoder.get(f"{branch}_fusion_method", "last_layer")
               for branch in ("condition", "future")}
    expected = {**methods, "fusion_layers": list(layers), **adafuse_feature_metadata(layers, dim, depth)}
    if (len(layers) < 2 or fusion != expected or "adafuse" not in methods.values()
            or any(m not in FUSION_METHODS or m == "concat_before_bottleneck" for m in methods.values())):
        raise ValueError("AdaFuse feature fusion config differs from the checkpoint manifest")
    with safe_open(source_dir / checkpoint_name, framework="np") as checkpoint:
        keys = set(checkpoint.keys())
        for branch in ("condition", "future"):
            prefix = f"video_backbone.{branch}_fuser."
            actual = {key for key in keys if key.startswith(prefix)}
            expected_shapes = ({prefix + "layer_logits": [len(layers), dim],
                                prefix + "post_fusion_norm.weight": [dim],
                                prefix + "post_fusion_norm.bias": [dim]}
                               if methods[f"{branch}_fusion_method"] == "adafuse" else {})
            if actual != set(expected_shapes):
                raise ValueError(f"AdaFuse checkpoint is missing or has unexpected {branch} fuser weights")
            for key, shape in expected_shapes.items():
                if checkpoint.get_slice(key).get_shape() != shape:
                    raise ValueError(f"AdaFuse checkpoint/config mismatch: {key}, expected {shape}")
    return fusion


def validate_siglip_checkpoint(source_dir, checkpoint_name, cfg):
    """Validate native headless SigLIP and the original M29 split projection."""
    vb = cfg.get("model", {}).get("video_backbone", {})
    encoder = vb.get("encoder") or {}
    if encoder.get("name") != "siglip":
        return None
    source_dir = Path(source_dir)
    hf = json.loads((source_dir/"siglip/config.json").read_text())
    settings = {k: encoder.get(k) for k in ("bottleneck_dim", "feature_norm_eps", "seperate_encode_view", "add_one_history_context")}
    if (not vb.get("from_scratch") or hf.get("vision_use_head") is not False
            or (hf.get("hidden_size"), hf.get("image_size"), hf.get("patch_size"), hf.get("num_hidden_layers")) != (1024, 256, 16, 24)
            or cfg.get("dataloader", {}).get("native_view_size") != 256
            or not settings["seperate_encode_view"] or not settings["add_one_history_context"]
            or json.loads((source_dir/"siglip/encoder.json").read_text()) != settings):
        raise ValueError("SigLIP native geometry/processing differs from its checkpoint")
    dit, enc = "video_backbone.dit.", "video_backbone.video_encoder._m."
    hidden = vb["dit_config"]["dim"]
    width = settings["bottleneck_dim"]
    expected = {
        enc+"embeddings.patch_embedding.weight": [1024, 3, 16, 16],
        enc+"embeddings.position_embedding.weight": [256, 1024],
        enc+"post_layernorm.weight": [1024],
        dit+"patch_embedding.clean_proj.weight": [hidden, 1024, 1, 2, 2],
        dit+"patch_embedding.proj1.weight": [width, 1024, 1, 2, 2],
        dit+"patch_embedding.proj2.weight": [hidden, width, 1, 1, 1],
        dit+"head.head.weight": [4096, hidden],
        dit+"head.head.bias": [4096],
        dit+"view_embedding.weight": [2, hidden],
    }
    with safe_open(source_dir/checkpoint_name, framework="np") as weights:
        keys = set(weights.keys())
        if any(k.startswith(enc+"head.") for k in keys):
            raise ValueError("SigLIP patch checkpoint unexpectedly contains a pooling head")
        for key, shape in expected.items():
            if key not in keys or weights.get_slice(key).get_shape() != shape:
                raise ValueError(f"SigLIP checkpoint mismatch: {key}, expected {shape}")
    return dict(latent_channels=1024, visual_tokens=1280, native_view_size=256)


def validate_dinov3_checkpoint(source_dir, checkpoint_name, cfg):
    """Check raw DINOv3 geometry and persisted processing before either benchmark."""
    vb = cfg.get("model", {}).get("video_backbone", {})
    encoder = vb.get("encoder") or {}
    if encoder.get("name") != "dinov3":
        return None
    source_dir = Path(source_dir)
    hf = json.loads((source_dir / "dinov3/config.json").read_text())
    if not vb.get("from_scratch"):
        raise ValueError("DINOv3 requires an external-encoder from_scratch checkpoint")
    if encoder.get("svae_path") is not None or encoder.get("svae_target_dim") is not None or (source_dir / "svae_config.json").exists():
        raise ValueError("DINOv3 LIBERO deployment currently requires raw features, without S-VAE")
    settings = {key: encoder.get(key, default) for key, default in {
        "use_avg_t_pooling": True, "add_one_history_context": False,
        "seperate_encode_view": False, "add_input_bottleneck": False,
        "bottleneck_dim": 512, "feature_norm_eps": 1e-6,
    }.items()}
    processing = source_dir / "dinov3/encoder.json"
    if processing.exists():
        if json.loads(processing.read_text()) != settings:
            raise ValueError("DINOv3 processing config differs from saved encoder.json")
    elif (not settings["use_avg_t_pooling"] or settings["add_one_history_context"]
          or settings["seperate_encode_view"] or settings["add_input_bottleneck"]):
        raise ValueError("Missing DINOv3 encoder.json processing contract")
    channels = int(hf["hidden_size"])
    if settings["seperate_encode_view"] and hf["patch_size"] != 16:
        raise ValueError("Independent DINOv3 views require patch_size=16")
    dit, enc = "video_backbone.dit.", "video_backbone.video_encoder._m."
    with safe_open(source_dir / checkpoint_name, framework="np") as weights:
        keys = set(weights.keys())
        hidden = weights.get_slice(dit + "head.head.weight").get_shape()[1]
        if hidden != (vb.get("dit_config") or {}).get("dim", hidden):
            raise ValueError("DiT hidden dimension differs from the saved config")
        expected = {
            enc + "norm.weight": [channels],
            dit + "head.head.weight": [4 * channels, hidden],
            dit + "head.head.bias": [4 * channels],
        }
        if settings["add_input_bottleneck"]:
            width = settings["bottleneck_dim"]
            expected.update({
                dit + "patch_embedding.clean_proj.weight": [hidden, channels, 1, 2, 2],
                dit + "patch_embedding.proj1.weight": [width, channels, 1, 2, 2],
                dit + "patch_embedding.proj2.weight": [hidden, width, 1, 1, 1],
            })
        else:
            expected[dit + "patch_embedding.weight"] = [hidden, channels, 1, 2, 2]
        if settings["seperate_encode_view"]:
            expected[dit + "view_embedding.weight"] = [2, hidden]
        for key, shape in expected.items():
            if key not in keys or weights.get_slice(key).get_shape() != shape:
                raise ValueError(f"DINOv3 checkpoint/config mismatch: {key}, expected {shape}")
    return dict(name="dinov3", latent_channels=channels, patch_size=[1, 2, 2],
                temporal_compression=4 if settings["use_avg_t_pooling"] else 1,
                num_condition_latents=2 if settings["add_one_history_context"] else 1)


