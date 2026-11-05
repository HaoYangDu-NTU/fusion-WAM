"""Train Route 2 with the historical FiLM-conditioned Action Expert."""

import argparse

from scripts.train_vjepa_policy import build_frozen_encoder, parameter_summary
from vjepa_policy.experimental.route2_policy import build_route2_baseline_policy
from vjepa_policy.models.wm_predictor import build_world_model_predictor
from vjepa_policy.train_common import (
    add_common_args,
    build_dataset_collator,
    build_patch_valid,
    get_model_image_size,
    get_model_video_frames,
    make_device,
    run_training,
    seed_process,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--activation-checkpointing-blocks", type=int, default=4)
    parser.add_argument(
        "--predictor-rope-frequency-pairing",
        choices=("corrected", "legacy"),
        default="corrected",
    )
    parser.add_argument(
        "--encoder-interpolate-rope",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def run(
    predictor_builder=build_world_model_predictor,
    policy_builder=build_route2_baseline_policy,
):
    args = parse_args()
    if args.action_chunk_size != args.num_frames - 1:
        raise ValueError("action_chunk_size must equal num_frames - 1")
    if args.pred_embed_dim % args.pred_num_heads:
        raise ValueError("pred_embed_dim must be divisible by pred_num_heads")
    if args.action_num_layers != args.pred_depth:
        raise ValueError("action_num_layers must equal pred_depth for direct K/V reuse")

    seed_process(args.seed)
    device = make_device()
    encoder = build_frozen_encoder(args)
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
        proprio_dim=args.proprio_dim,
        use_activation_checkpointing=args.use_activation_checkpointing,
        activation_checkpointing_blocks=args.activation_checkpointing_blocks,
        interpolate_rope=True,
        corrected_rope_frequency_pairing=(
            args.predictor_rope_frequency_pairing == "corrected"
        ),
    )
    image_height, image_width = get_model_image_size(args)
    context_len = args.context_tubelets * (
        (image_height // args.patch_size) * (image_width // args.patch_size)
    )
    model = policy_builder(
        encoder,
        predictor,
        action_dim=args.action_dim,
        action_chunk_size=args.action_chunk_size,
        context_len=context_len,
        proprio_dim=args.proprio_dim,
        action_hidden_size=args.action_hidden_size,
        action_num_layers=args.action_num_layers,
        action_num_inference_steps=args.action_num_inference_steps,
        loss_exp=args.loss_exp,
        action_loss_weight=args.action_loss_weight,
        patch_valid=build_patch_valid(args),
    )
    parameter_summary(model)
    model = model.to(device)
    dataset, collator = build_dataset_collator(
        args, action_chunk_size=args.action_chunk_size
    )
    run_training(args, model, dataset, collator)


if __name__ == "__main__":
    run()
