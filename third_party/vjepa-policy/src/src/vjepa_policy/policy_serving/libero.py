"""LIBERO inference adapter for the published V-JEPA policy."""

import hashlib
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from vjepa_policy.data import CausalPatchMask
from vjepa_policy.datasets import (
    DEFAULT_PROMPT,
    ActionStateNormalizer,
    QuadrantViewCombiner,
    VideoClipTransform,
    combine_video_clips,
)
from vjepa_policy.models.wm_predictor import (
    build_world_model_predictor,
)
from vjepa_policy.models.vjepa_policy import (
    build_vjepa_policy,
    encode_video_views,
    encoded_proprio_dim,
)
from vjepa_policy.experimental.context_only_policy import build_context_only_policy
from vjepa_policy.state import packed_state_dim, prepare_proprio_state
from vjepa_policy.models import vision_transformer as video_vit
from vjepa_policy.models.vjepa2_1 import vision_transformer as video_vit_2_1


_TOPOLOGY_CLASS_ALIASES = {
    "FreshLargeCrossAttentionPredictor": "WorldModelPredictor",
    "LargePredictorActionCrossExpert": "ActionExpert",
    "CleanContextLargeCrossAttentionPredictor": "CleanContextWorldModelPredictor",
}

logger = logging.getLogger(__name__)


def _canonicalize_topology_value(value):
    return _TOPOLOGY_CLASS_ALIASES.get(value, value)


@dataclass(frozen=True)
class PolicyServingConfig:
    precision: str = "bf16"
    encoder_family: str = "vjepa2"
    encoder_checkpoint_key: str = "encoder"
    model_name: str = "vit_giant_xformers"
    crop_size: int = 224
    image_resize_mode: str = "stretch"
    patch_size: int = 16
    num_frames: int = 10
    tubelet_size: int = 2
    context_tubelets: int = 1
    pred_depth: int = 24
    pred_embed_dim: int = 1024
    pred_num_heads: int = 16
    num_mask_tokens: int = 10
    lang_dim: int = 4096
    t5_len: int = 32
    action_dim: int = 7
    proprio_dim: int = 8
    # ``None`` lets serving recover the fixed-width contract from checkpoint
    # metadata; absent metadata retains the historical variable-width path.
    max_state_dim: int | None = None
    proprio_encoding: str = "identity"
    action_chunk_size: int = 32
    action_hidden_size: int = 512
    action_num_layers: int = 24
    action_num_inference_steps: int = 10
    condition_num_heads: int = 8
    corrected_predictor_rope: bool = True
    encoder_interpolate_rope: bool = True
    view_layout: str = "independent"
    camera_keys: tuple[str, ...] = (
        "observation.images.image",
        "observation.images.wrist_image",
    )
    dataset_stats: str | None = None
    context_only: bool = False

    @classmethod
    def from_checkpoint(cls, ckpt_path, overrides=None):
        """Recover saved serving settings, then apply explicit overrides."""
        checkpoint = torch.load(
            ckpt_path, map_location="cpu", weights_only=False, mmap=True
        )
        topology = checkpoint.get("model_topology") or {}
        saved = checkpoint.get("policy_serving") or {}
        values = {}

        train_precision = checkpoint.get("train_precision")
        if train_precision in ("bf16", "fp16"):
            values["precision"] = train_precision
        elif train_precision == "no":
            values["precision"] = "fp32"

        topology_fields = {
            "encoder_family": "encoder_family",
            "encoder_model_name": "model_name",
            "encoder_checkpoint_key": "encoder_checkpoint_key",
            "predictor_image_height": "crop_size",
            "image_resize_mode": "image_resize_mode",
            "predictor_depth": "pred_depth",
            "predictor_embed_dim": "pred_embed_dim",
            "predictor_num_heads": "pred_num_heads",
            "view_layout": "view_layout",
            "camera_keys": "camera_keys",
            "action_dim": "action_dim",
            "raw_proprio_dim": "proprio_dim",
            "max_state_dim": "max_state_dim",
            "proprio_encoding": "proprio_encoding",
            "action_chunk_size": "action_chunk_size",
            "action_hidden_size": "action_hidden_size",
            "action_num_layers": "action_num_layers",
            "encoder_interpolate_rope": "encoder_interpolate_rope",
        }
        for source, target in topology_fields.items():
            value = topology.get(source)
            if value is not None:
                values[target] = tuple(value) if target == "camera_keys" else value
        rope_pairing = topology.get("predictor_rope_frequency_pairing")
        if rope_pairing is not None:
            values["corrected_predictor_rope"] = rope_pairing == "corrected"

        saved_config = saved.get("config") if isinstance(saved, dict) else None
        if saved_config:
            valid_fields = {field.name for field in fields(cls)}
            for name, value in saved_config.items():
                if name not in valid_fields:
                    continue
                # An explicit topology field is more informative than a
                # legacy serving blob that serialized it as null.
                if name == "max_state_dim" and value is None:
                    continue
                values[name] = tuple(value) if name == "camera_keys" else value
        if overrides:
            values.update(overrides)
        del checkpoint
        return cls(**values)


def _strip_prefix(state_dict, prefix="module.backbone."):
    return {
        name[len(prefix) :]: value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }


def to_env_action(action, binarize_gripper=True):
    """Convert LeRobot LIBERO actions to the robosuite action convention."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 1.0 - 2.0 * action[..., -1]
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    action[..., :6] = np.clip(action[..., :6], -1.0, 1.0)
    return action


class VJEPAPolicyServing:
    def __init__(
        self,
        ckpt_path,
        pretrained_encoder,
        text_cache_dir,
        config=None,
        device="cuda",
        encoder_builder=None,
        predictor_builder=None,
        policy_builder=None,
    ):
        if config is None:
            args = PolicyServingConfig.from_checkpoint(ckpt_path)
        else:
            args = config
            metadata = PolicyServingConfig.from_checkpoint(ckpt_path)
            if metadata.context_only:
                args = replace(args, context_only=True)
            # A caller may provide the ordinary LIBERO settings while leaving
            # the state contract unspecified.  Recover only that contract so
            # all other explicit serving settings remain untouched.
            if args.max_state_dim is None:
                if metadata.max_state_dim is not None:
                    args = replace(args, max_state_dim=metadata.max_state_dim)
        if args.max_state_dim == 0:
            args = replace(args, max_state_dim=None)
        self.device = torch.device(device)
        self.context_only = bool(args.context_only)
        self.precision = args.precision
        precision_dtypes = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if self.precision not in precision_dtypes:
            raise ValueError(
                f"unsupported precision={self.precision!r}; choose bf16, fp16, or fp32"
            )
        self.dtype = precision_dtypes[self.precision]
        if self.precision == "fp32":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.crop_size = args.crop_size
        self.t5_len = args.t5_len
        self.binarize_gripper = True
        if args.view_layout not in ("quadrant", "horizontal", "independent"):
            raise ValueError(f"unsupported serving view_layout={args.view_layout!r}")
        self.view_layout = args.view_layout
        if not args.camera_keys:
            raise ValueError("camera_keys must contain at least one view")
        self.camera_keys = tuple(args.camera_keys)
        if args.view_layout == "quadrant" and len(self.camera_keys) > 4:
            raise ValueError("quadrant serving supports at most four camera views")
        if args.view_layout == "horizontal" and len(self.camera_keys) < 2:
            raise ValueError("horizontal serving requires at least two camera views")
        num_views = len(self.camera_keys) if args.view_layout == "independent" else 1
        image_size = (
            (args.crop_size, args.crop_size * len(self.camera_keys))
            if args.view_layout == "horizontal"
            else (args.crop_size, args.crop_size)
        )
        self.model_image_size = image_size
        policy_factory = policy_builder or (
            build_context_only_policy if args.context_only else build_vjepa_policy
        )
        state_contract_max = (
            args.max_state_dim if policy_factory is build_vjepa_policy else None
        )
        if state_contract_max is None:
            model_proprio_dim = encoded_proprio_dim(
                args.proprio_dim, args.proprio_encoding
            )
        else:
            if args.proprio_encoding != "identity":
                raise ValueError(
                    "fixed-width state packing requires proprio_encoding='identity'"
                )
            if args.proprio_dim > state_contract_max:
                raise ValueError(
                    f"proprio_dim={args.proprio_dim} exceeds "
                    f"max_state_dim={state_contract_max}"
                )
            model_proprio_dim = packed_state_dim(state_contract_max)

        if encoder_builder is None:
            if args.encoder_family not in ("vjepa2", "vjepa2_1"):
                raise ValueError(f"unsupported encoder_family={args.encoder_family!r}")
            if args.encoder_family == "vjepa2" and args.model_name == "vit_giant":
                raise ValueError(
                    "Use vit_giant_xformers for the official 22-head V-JEPA2 ViT-G"
                )
            encoder_module = (
                video_vit if args.encoder_family == "vjepa2" else video_vit_2_1
            )
            encoder_kwargs = dict(
                img_size=image_size,
                patch_size=args.patch_size,
                num_frames=args.num_frames,
                tubelet_size=args.tubelet_size,
                uniform_power=False,
                use_rope=True,
                use_sdpa=True,
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
            if (
                args.model_name == "vit_giant_xformers"
                and encoder.blocks[0].attn.num_heads != 22
            ):
                raise RuntimeError(
                    "Official V-JEPA2 ViT-G must have 22 attention heads"
                )
        else:
            encoder = encoder_builder(args, pretrained_encoder)
        predictor_factory = predictor_builder or build_world_model_predictor
        predictor = predictor_factory(
            img_size=image_size,
            patch_size=args.patch_size,
            num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=args.pred_embed_dim,
            depth=args.pred_depth,
            num_heads=args.pred_num_heads,
            num_mask_tokens=args.num_mask_tokens,
            lang_dim=args.lang_dim,
            proprio_dim=model_proprio_dim,
            use_activation_checkpointing=False,
            interpolate_rope=True,
            corrected_rope_frequency_pairing=args.corrected_predictor_rope,
            num_views=num_views,
        )
        mask_builder = getattr(encoder, "build_policy_mask", None)
        if mask_builder is None:
            mask = CausalPatchMask(
                image_size=image_size,
                patch_size=args.patch_size,
                tubelet_size=args.tubelet_size,
                video_frames=args.num_frames,
                context_tubelets=args.context_tubelets,
                num_views=num_views,
            )
        else:
            mask = mask_builder(image_size=image_size, num_views=num_views)
        policy_kwargs = dict(
            action_dim=args.action_dim,
            action_chunk_size=args.action_chunk_size,
            context_len=mask.n_ctx,
            proprio_dim=args.proprio_dim,
            proprio_encoding=args.proprio_encoding,
            action_hidden_size=args.action_hidden_size,
            action_num_layers=args.action_num_layers,
            condition_num_heads=args.condition_num_heads,
            action_num_inference_steps=args.action_num_inference_steps,
        )
        if policy_factory is build_vjepa_policy:
            policy_kwargs["max_state_dim"] = state_contract_max
        policy = policy_factory(encoder, predictor, **policy_kwargs)

        if encoder_builder is None:
            encoder_checkpoint = torch.load(
                pretrained_encoder, map_location="cpu", weights_only=False, mmap=True
            )
            if args.encoder_checkpoint_key not in encoder_checkpoint:
                raise KeyError(
                    f"Encoder checkpoint has no {args.encoder_checkpoint_key!r}"
                )
            print(
                "[load] encoder:",
                policy.encoder.load_state_dict(
                    _strip_prefix(encoder_checkpoint[args.encoder_checkpoint_key])
                ),
            )
            del encoder_checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        expected_precision = {
            "fp32": "no",
            "fp16": "fp16",
            "bf16": "bf16",
        }[self.precision]
        checkpoint_precision = checkpoint.get("train_precision")
        if checkpoint_precision is None:
            logger.warning(
                "Checkpoint has no train_precision metadata; relying on state-dict validation"
            )
        elif checkpoint_precision != expected_precision:
            raise ValueError(
                f"checkpoint precision {checkpoint_precision!r} "
                f"!= serving {expected_precision!r}"
            )
        topology = checkpoint.get("model_topology", {})
        encoder_attention = policy.encoder.blocks[0].attn
        encoder_family = getattr(
            policy.encoder, "_vjepa_policy_family", args.encoder_family
        )
        encoder_model_name = getattr(
            policy.encoder, "_vjepa_policy_model_name", args.model_name
        )
        encoder_checkpoint_key = getattr(
            policy.encoder,
            "_vjepa_policy_checkpoint_key",
            args.encoder_checkpoint_key,
        )
        expected_topology = {
            "predictor_class": type(policy.predictor).__name__,
            "predictor_depth": args.pred_depth,
            "predictor_embed_dim": args.pred_embed_dim,
            "predictor_num_heads": args.pred_num_heads,
            "predictor_image_height": image_size[0],
            "predictor_image_width": image_size[1],
            "predictor_num_views": num_views,
            "action_expert_class": type(policy.action_expert).__name__,
            "action_hidden_size": args.action_hidden_size,
            "action_num_layers": args.action_num_layers,
            "action_num_heads": args.pred_num_heads,
            "action_head_dim": args.pred_embed_dim // args.pred_num_heads,
            "action_dim": args.action_dim,
            "action_chunk_size": args.action_chunk_size,
            "proprio_encoding": args.proprio_encoding,
            "raw_proprio_dim": args.proprio_dim,
            "encoded_proprio_dim": encoded_proprio_dim(
                args.proprio_dim, args.proprio_encoding
            ),
            "max_state_dim": state_contract_max or 0,
            "packed_proprio_dim": model_proprio_dim,
            "predictor_interpolate_rope": True,
            "predictor_rope_frequency_pairing": (
                "corrected" if args.corrected_predictor_rope else "legacy"
            ),
            "encoder_interpolate_rope": bool(
                getattr(encoder_attention, "interpolate_rope", False)
            ),
            "encoder_rope_frequency_pairing": (
                "corrected"
                if getattr(encoder_attention, "corrected_frequency_pairing", False)
                else "legacy"
            ),
            "clean_context_attention": bool(
                getattr(policy.predictor, "use_clean_context_attention", False)
            ),
            "read_full_predictor_kv": bool(
                getattr(policy.action_expert, "read_full_predictor_kv", False)
            ),
        }
        missing_topology_defaults = {
            "predictor_interpolate_rope": True,
            "predictor_rope_frequency_pairing": "legacy",
            "encoder_interpolate_rope": False,
            "encoder_rope_frequency_pairing": "legacy",
            "predictor_image_height": 256,
            "predictor_image_width": 256,
            "predictor_num_views": 1,
            "action_dim": args.action_dim,
            "action_chunk_size": args.action_chunk_size,
            "proprio_encoding": "identity",
            "raw_proprio_dim": args.proprio_dim,
            "encoded_proprio_dim": args.proprio_dim,
            "max_state_dim": 0,
            "packed_proprio_dim": args.proprio_dim,
            "clean_context_attention": False,
            "read_full_predictor_kv": False,
        }
        if topology:
            observed_topology = {}
            for name in expected_topology:
                if name in topology:
                    observed_topology[name] = _canonicalize_topology_value(
                        topology[name]
                    )
                elif name in missing_topology_defaults:
                    observed_topology[name] = _canonicalize_topology_value(
                        missing_topology_defaults[name]
                    )
            mismatches = {
                name: (observed_topology[name], expected)
                for name, expected in expected_topology.items()
                if name in observed_topology
                if observed_topology[name] != expected
            }
            if mismatches:
                raise ValueError(f"checkpoint topology mismatch: {mismatches}")
            missing_topology = sorted(set(expected_topology) - set(observed_topology))
            if missing_topology:
                logger.warning(
                    "Checkpoint topology omits %s; relying on state-dict validation",
                    missing_topology,
                )
        else:
            logger.warning(
                "Checkpoint has no model_topology metadata; relying on state-dict validation"
            )
        optional_topology = {
            "view_layout": args.view_layout,
            "camera_keys": list(self.camera_keys),
            "encoder_family": encoder_family,
            "encoder_model_name": encoder_model_name,
            "encoder_checkpoint_key": encoder_checkpoint_key,
        }
        optional_mismatches = {
            name: (topology[name], expected)
            for name, expected in optional_topology.items()
            if name in topology and topology[name] != expected
        }
        if optional_mismatches:
            raise ValueError(
                f"checkpoint input/encoder mismatch: {optional_mismatches}"
            )
        print(
            "[load] predictor:",
            policy.predictor.load_state_dict(checkpoint["predictor"]),
        )
        print(
            "[load] action_expert:",
            policy.action_expert.load_state_dict(checkpoint["action_expert"]),
        )
        print(
            "[load] proprio_encoder:",
            policy.proprio_encoder.load_state_dict(checkpoint["proprio_encoder"]),
        )
        print("[load] checkpoint step:", checkpoint.get("step"))
        del checkpoint

        self.encoder = policy.encoder.to(self.device).eval()
        self.predictor = policy.predictor.to(self.device).eval()
        self.action_expert = policy.action_expert.to(self.device).eval()
        self.proprio_encoder = policy.proprio_encoder.to(self.device).eval()
        self.context_len = mask.n_ctx
        self.num_frames = args.num_frames
        self.action_chunk_size = args.action_chunk_size
        self.raw_proprio_dim = args.proprio_dim
        self.max_state_dim = state_contract_max
        self.proprio_encoding = args.proprio_encoding
        self.m_enc = mask.ctx_idx.unsqueeze(0).to(self.device)
        self.m_pred = mask.tgt_idx.unsqueeze(0).to(self.device)

        if args.view_layout == "independent":
            self.video_transform = VideoClipTransform(
                (args.crop_size, args.crop_size), resize_mode=args.image_resize_mode
            )
            self.view_combiner = "independent"
        elif args.view_layout == "horizontal":
            self.video_transform = VideoClipTransform(
                (args.crop_size, args.crop_size), resize_mode=args.image_resize_mode
            )
            self.view_combiner = "horizontal"
        else:
            self.video_transform = VideoClipTransform(
                (args.crop_size // 2, args.crop_size // 2),
                resize_mode=args.image_resize_mode,
            )
            self.view_combiner = QuadrantViewCombiner(
                image_size, fill_value=self.video_transform.normalized_black
            )
        self.text_cache_dir = text_cache_dir
        self._lang_cache = {}
        stats_path = args.dataset_stats or os.path.join(
            os.path.dirname(ckpt_path), "dataset_stats.json"
        )
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(
                f"action normalization stats not found: {stats_path}"
            )
        self.action_normalizer = ActionStateNormalizer.load(stats_path)

    def _load_language(self, task_text):
        if task_text in self._lang_cache:
            return self._lang_cache[task_text]
        prompt = DEFAULT_PROMPT.format(task=task_text)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = os.path.join(self.text_cache_dir, f"{digest}.t5_len{self.t5_len}.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        language = payload["context"].unsqueeze(0).to(self.device)
        language_mask = payload["mask"].unsqueeze(0).bool().to(self.device)
        self._lang_cache[task_text] = (language, language_mask)
        return language, language_mask

    @staticmethod
    def _to_uint8_hwc(image):
        if image.ndim == 4:
            image = image.squeeze(0)
        image = (image.float() * 255.0).clamp(0, 255).round().to(torch.uint8)
        return image.permute(1, 2, 0).cpu().numpy()

    @staticmethod
    def _to_state_tensor(state):
        state = state if isinstance(state, torch.Tensor) else torch.as_tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(
                f"observation.state must be [D] or [B,D], got {tuple(state.shape)}"
            )
        return state.float()

    def _prepare_observation(self, obs):
        views = []
        for key in self.camera_keys:
            current = self._to_uint8_hwc(obs[key])
            past_key = f"{key}_past"
            if past_key not in obs:
                raise KeyError(f"Missing past-frame view {past_key!r}")
            past = self._to_uint8_hwc(obs[past_key])
            frames = np.stack([past, current] + [current] * (self.num_frames - 2))
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
            views.append(self.video_transform(frames))
        clip = combine_video_clips(views, self.view_combiner)

        language, language_mask = self._load_language(obs["task"][0])
        state_key = "observation.state"
        if state_key not in obs:
            raise KeyError(f"Missing {state_key!r}; proprio conditioning is required")
        state = self._to_state_tensor(obs[state_key])
        return clip, language, language_mask, state

    @torch.no_grad()
    def infer_batch(self, observations):
        if not observations:
            return []
        prepared = [self._prepare_observation(obs) for obs in observations]
        clips = torch.stack([item[0] for item in prepared]).to(self.device)
        language = torch.cat([item[1] for item in prepared])
        language_mask = torch.cat([item[2] for item in prepared])
        state_key = "observation.state"
        state = torch.cat([item[3] for item in prepared])
        state = self.action_normalizer({state_key: state})[state_key].to(self.device)
        state = prepare_proprio_state(
            state,
            raw_state_dim=getattr(self, "raw_proprio_dim", state.shape[-1]),
            encoding=getattr(self, "proprio_encoding", "identity"),
            max_state_dim=getattr(self, "max_state_dim", None),
        )
        batch_size = len(observations)
        m_enc = self.m_enc.expand(batch_size, -1)
        m_pred = self.m_pred.expand(batch_size, -1)
        precision_context = (
            torch.autocast(device_type=self.device.type, dtype=self.dtype)
            if self.precision != "fp32"
            else nullcontext()
        )
        with precision_context:
            context_latents = encode_video_views(self.encoder, clips, m_enc)
            if self.context_only:
                predictor_kv = self.predictor.forward_context_only(
                    context_latents,
                    m_enc,
                    language=language,
                    language_mask=language_mask,
                    proprio=state,
                )
            else:
                _, predictor_kv = self.predictor(
                    context_latents,
                    m_enc,
                    m_pred,
                    language=language,
                    language_mask=language_mask,
                    proprio=state,
                )
            actions = self.action_expert.predict_action(
                predictor_kv,
                language,
                language_mask,
                state,
                context_len=self.context_len,
            )
        actions = self.action_normalizer.unnormalize_action(actions.float().cpu())
        return [
            {
                "actions": to_env_action(
                    action.numpy(), binarize_gripper=self.binarize_gripper
                )
            }
            for action in actions
        ]

    def infer(self, obs):
        return self.infer_batch([obs])[0]
