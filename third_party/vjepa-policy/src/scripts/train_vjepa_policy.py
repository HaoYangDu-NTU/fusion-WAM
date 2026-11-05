"""Train the language-conditioned V-JEPA world model and Action Expert."""

import argparse

import torch

from vjepa_policy.models.wm_predictor import (
    build_world_model_predictor,
)
from vjepa_policy.models.vjepa_policy import (
    build_vjepa_policy,
    encoded_proprio_dim,
)
from vjepa_policy.state import packed_state_dim
from vjepa_policy.models import vision_transformer as video_vit
from vjepa_policy.models.vjepa2_1 import vision_transformer as video_vit_2_1
from vjepa_policy.train_common import (
    _strip_prefix,
    add_common_args,
    build_dataset_collator,
    build_patch_valid,
    get_model_image_size,
    get_model_video_frames,
    get_camera_keys,
    make_device,
    run_training,
    seed_process,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train V-JEPA latent prediction and flow-matching action generation."
    )
    add_common_args(parser)
    parser.set_defaults(
        pred_depth=24,
        pred_embed_dim=1024,
        pred_num_heads=16,
        action_normalization="QUANTILE",
        state_normalization="QUANTILE",
        batch_size=32,
        num_workers=8,
        prefetch_factor=2,
        weight_decay=0.01,
        num_epochs=10,
        max_steps=21360,
        mixed_precision="bf16",
        seed=7,
    )
    parser.add_argument("--action-chunk-size", type=int, default=32)
    parser.add_argument("--action-hidden-size", type=int, default=512)
    parser.add_argument("--action-num-layers", type=int, default=24)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-num-inference-steps", type=int, default=10)
    parser.add_argument("--condition-num-heads", type=int, default=8)
    parser.add_argument("--activation-checkpointing-blocks", type=int, default=12)
    parser.add_argument(
        "--predictor-rope-frequency-pairing",
        choices=("corrected", "legacy"),
        default="corrected",
    )
    parser.add_argument(
        "--encoder-interpolate-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--predictor-view-map",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional target-to-source row map for predictor view embeddings. "
            "All other predictor parameters still load strictly."
        ),
    )
    return parser.parse_args(argv)


def build_frozen_encoder(args):
    if args.encoder_family == "vjepa2" and args.model_name == "vit_giant":
        raise ValueError("Use vit_giant_xformers for the official 22-head V-JEPA2 ViT-G")
    encoder_module = video_vit if args.encoder_family == "vjepa2" else video_vit_2_1
    encoder_kwargs = dict(
        img_size=get_model_image_size(args),
        patch_size=args.patch_size,
        num_frames=get_model_video_frames(args),
        tubelet_size=args.tubelet_size,
        uniform_power=False,
        use_rope=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        interpolate_rope=args.encoder_interpolate_rope,
    )
    if args.encoder_family == "vjepa2_1":
        encoder_kwargs.update(
            img_temporal_dim_size=1,
            modality_embedding=True,
            n_output_distillation=1,
        )
    encoder = encoder_module.__dict__[args.model_name](**encoder_kwargs)
    encoder._vjepa_policy_training_false = args.encoder_family == "vjepa2_1"
    if args.model_name == "vit_giant_xformers" and encoder.blocks[0].attn.num_heads != 22:
        raise RuntimeError("Official ViT-G must have 22 attention heads")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if args.encoder_checkpoint_key not in checkpoint:
        raise KeyError(
            f"Checkpoint has no {args.encoder_checkpoint_key!r}; available keys: "
            f"{sorted(checkpoint)}"
        )
    print(
        f"[load] {args.encoder_family}/{args.model_name} "
        f"from {args.encoder_checkpoint_key}:",
        encoder.load_state_dict(
            _strip_prefix(checkpoint[args.encoder_checkpoint_key]), strict=True
        ),
    )
    del checkpoint
    return encoder


def parameter_summary(model):
    groups = {
        "encoder_frozen": model.encoder,
        "predictor": model.predictor,
        "action_expert": model.action_expert,
        "proprio_encoder": model.proprio_encoder,
    }
    for name, module in groups.items():
        count = sum(parameter.numel() for parameter in module.parameters())
        trainable = sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        )
        print(f"[parameters] {name}={count:,} ({count / 1e6:.2f}M), trainable={trainable:,}")


def _load_predictor_initialization(predictor, checkpoint_path, view_map=None):
    """Load only a predictor state dict; never restore training progress."""
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if isinstance(payload, dict) and "predictor" in payload:
        state = payload["predictor"]
        metadata = payload.get("model_topology") or {}
    elif isinstance(payload, dict) and all(torch.is_tensor(value) for value in payload.values()):
        state = payload
        metadata = {}
    else:
        raise ValueError(
            f"{checkpoint_path} is not a predictor state dict or policy checkpoint"
        )
    # DDP exports may retain a module prefix; the policy checkpoint normally
    # contains predictor-relative keys already.
    if state and all(name.startswith("module.") for name in state):
        state = {name[len("module.") :]: value for name, value in state.items()}
    metadata = dict(metadata)
    if view_map is not None:
        key = "view_embedding.weight"
        if key not in state or predictor.view_embedding is None:
            raise ValueError(
                "--predictor-view-map requires view embeddings in both source and target"
            )
        source = state[key]
        target_rows = predictor.view_embedding.num_embeddings
        if len(view_map) != target_rows:
            raise ValueError(
                f"predictor view map needs {target_rows} entries, got {len(view_map)}"
            )
        invalid = [index for index in view_map if not 0 <= index < source.shape[0]]
        if invalid:
            raise ValueError(
                f"predictor view map indices {invalid} exceed source rows={source.shape[0]}"
            )
        source_rows = torch.tensor(view_map, dtype=torch.long)
        state = dict(state)
        state[key] = source.index_select(0, source_rows)
        metadata["view_embedding_initialization"] = {
            "source_rows": int(source.shape[0]),
            "target_rows": int(target_rows),
            "target_to_source_map": list(view_map),
        }
    result = predictor.load_state_dict(state, strict=True)
    print(f"[load] predictor initialization from {checkpoint_path}: {result}")
    return metadata


def main(
    policy_builder=build_vjepa_policy,
    encoder_builder=build_frozen_encoder,
    predictor_builder=build_world_model_predictor,
    dataset_collator_builder=build_dataset_collator,
    args=None,
):
    args = parse_args() if args is None else args
    if args.action_chunk_size != args.num_frames - 1:
        raise ValueError("action_chunk_size must equal num_frames - 1")
    if args.pred_embed_dim % args.pred_num_heads:
        raise ValueError("pred_embed_dim must be divisible by pred_num_heads")
    if args.action_num_layers != args.pred_depth:
        raise ValueError("action_num_layers must equal pred_depth for direct K/V reuse")
    if args.action_hidden_size % args.condition_num_heads:
        raise ValueError("action_hidden_size must be divisible by condition_num_heads")
    predictor_init = getattr(args, "predictor_init", None)
    resume = getattr(args, "resume", None)
    max_state_dim = int(getattr(args, "max_state_dim", 0) or 0)
    if predictor_init and resume:
        raise ValueError("--predictor-init cannot be combined with --resume")

    seed_process(args.seed)
    device = make_device()
    camera_keys = get_camera_keys(args)
    num_views = len(camera_keys) if args.view_layout == "independent" else 1
    encoder = encoder_builder(args)
    use_fixed_state = bool(max_state_dim) and policy_builder is build_vjepa_policy
    if use_fixed_state:
        if max_state_dim <= 0:
            raise ValueError("max_state_dim must be positive or 0")
        if args.proprio_encoding != "identity":
            raise ValueError(
                "fixed-width state packing requires proprio_encoding='identity'; "
                "use --max-state-dim 0 for legacy conditioning"
            )
        if args.proprio_dim > max_state_dim:
            raise ValueError(
                f"proprio_dim={args.proprio_dim} exceeds max_state_dim={max_state_dim}"
            )
        model_proprio_dim = packed_state_dim(max_state_dim)
    else:
        model_proprio_dim = encoded_proprio_dim(
            args.proprio_dim, args.proprio_encoding
        )
    predictor = predictor_builder(
        img_size=get_model_image_size(args),
        patch_size=args.patch_size,
        num_frames=get_model_video_frames(args),
        tubelet_size=args.tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=args.pred_embed_dim,
        depth=args.pred_depth,
        num_heads=args.pred_num_heads,
        num_mask_tokens=args.num_mask_tokens,
        lang_dim=args.lang_dim,
        proprio_dim=model_proprio_dim,
        use_activation_checkpointing=args.use_activation_checkpointing,
        activation_checkpointing_blocks=args.activation_checkpointing_blocks,
        interpolate_rope=True,
        corrected_rope_frequency_pairing=(
            args.predictor_rope_frequency_pairing == "corrected"
        ),
        num_views=num_views,
    )
    image_height, image_width = get_model_image_size(args)
    context_len = args.context_tubelets * (
        num_views
        * (image_height // args.patch_size)
        * (image_width // args.patch_size)
    )
    policy_kwargs = dict(
        action_dim=args.action_dim,
        action_chunk_size=args.action_chunk_size,
        context_len=context_len,
        proprio_dim=args.proprio_dim,
        proprio_encoding=args.proprio_encoding,
        action_hidden_size=args.action_hidden_size,
        action_num_layers=args.action_num_layers,
        condition_num_heads=args.condition_num_heads,
        action_num_inference_steps=args.action_num_inference_steps,
        loss_exp=args.loss_exp,
        action_loss_weight=args.action_loss_weight,
        patch_valid=build_patch_valid(args),
    )
    if policy_builder is build_vjepa_policy:
        policy_kwargs["max_state_dim"] = max_state_dim if use_fixed_state else None
    model = policy_builder(encoder, predictor, **policy_kwargs)
    predictor_init_topology = None
    if predictor_init:
        if policy_builder is not build_vjepa_policy:
            raise ValueError("--predictor-init is supported only by the canonical policy builder")
        predictor_init_topology = _load_predictor_initialization(
            model.predictor,
            predictor_init,
            view_map=getattr(args, "predictor_view_map", None),
        )
    model.camera_keys = tuple(camera_keys)
    model.view_layout = args.view_layout
    model.encoder_family = getattr(
        encoder, "_vjepa_policy_family", args.encoder_family
    )
    model.encoder_model_name = getattr(
        encoder, "_vjepa_policy_model_name", args.model_name
    )
    model.encoder_checkpoint_key = getattr(
        encoder, "_vjepa_policy_checkpoint_key", args.encoder_checkpoint_key
    )
    model.action_indices = tuple(args.action_indices) if args.action_indices else None
    model.state_indices = tuple(args.state_indices) if args.state_indices else None
    model.relative_action_indices = (
        tuple(args.relative_action_indices) if args.relative_action_indices else None
    )
    model.instruction_field = args.instruction_field
    model.image_resize_mode = args.image_resize_mode
    model.policy_serving = {
        "config": {
            "precision": "fp32"
            if args.mixed_precision == "no"
            else args.mixed_precision,
            "encoder_family": model.encoder_family,
            "encoder_checkpoint_key": model.encoder_checkpoint_key,
            "model_name": model.encoder_model_name,
            "crop_size": args.crop_size,
            "image_resize_mode": args.image_resize_mode,
            "patch_size": args.patch_size,
            "num_frames": get_model_video_frames(args),
            "tubelet_size": args.tubelet_size,
            "context_tubelets": args.context_tubelets,
            "pred_depth": args.pred_depth,
            "pred_embed_dim": args.pred_embed_dim,
            "pred_num_heads": args.pred_num_heads,
            "num_mask_tokens": args.num_mask_tokens,
            "lang_dim": args.lang_dim,
            "t5_len": args.context_len,
            "action_dim": args.action_dim,
            "proprio_dim": args.proprio_dim,
            "max_state_dim": max_state_dim if use_fixed_state else 0,
            "packed_proprio_dim": model_proprio_dim,
            "proprio_encoding": args.proprio_encoding,
            "action_chunk_size": args.action_chunk_size,
            "action_hidden_size": args.action_hidden_size,
            "action_num_layers": args.action_num_layers,
            "action_num_inference_steps": args.action_num_inference_steps,
            "condition_num_heads": args.condition_num_heads,
            "corrected_predictor_rope": (
                args.predictor_rope_frequency_pairing == "corrected"
            ),
            "encoder_interpolate_rope": args.encoder_interpolate_rope,
            "view_layout": args.view_layout,
            "camera_keys": list(camera_keys),
            "context_only": bool(getattr(model, "context_only", False)),
        },
        "encoder_adapter": getattr(
            args,
            "geometry_experiment",
            getattr(args, "latent_encoder", "builtin"),
        ),
    }
    if predictor_init_topology is not None:
        model.predictor_initialization = {
            "path": predictor_init,
            "source_topology": predictor_init_topology,
        }
    parameter_summary(model)
    model = model.to(device)
    dataset, collator = dataset_collator_builder(
        args, action_chunk_size=args.action_chunk_size
    )
    run_training(args, model, dataset, collator)


if __name__ == "__main__":
    main()
