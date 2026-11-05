"""DualSystem joint self-attention architecture.

True joint attention (MMDiT / FastWAM MoT style): at every transformer
layer, the video and action backbones each compute Q/K/V independently
through ``pre_attn_at_layer``; :class:`DualSystemMoTDriver` concatenates the
two modalities, runs a single mixed self-attention, splits the result,
and feeds each slice back through ``post_attn_at_layer``.

The driver lives on ``self._mot_driver`` as a plain Python object with
no parameters; ``forward`` delegates the per-layer loop to it.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO
from openwam.model.architectures.utils.action_router import ActionVisualRouter
from openwam.model.compile_options import (
    compile_enabled,
    section_enabled,
    self_attn_compile_cfg,
    torch_compile_kwargs,
)

from openwam.train.utils.step_timing import timed

logger = logging.getLogger(__name__)


def _fixed_wan_text_context(context, mask=None, seq_lens=None):
    """Use exactly 128 text slots; proprio is appended separately afterward."""
    if mask is None:
        positions = torch.arange(context.shape[1], device=context.device)
        lengths = context.shape[1] if seq_lens is None else seq_lens.to(context.device).unsqueeze(1)
        mask = (positions.unsqueeze(0) < lengths).expand(context.shape[0], -1)
    else:
        mask = mask.to(device=context.device, dtype=torch.bool)
    context, mask = context[:, :128], mask[:, :128]
    padding = 128 - context.shape[1]
    if padding:
        context = torch.nn.functional.pad(context, (0, 0, 0, padding))
        mask = torch.nn.functional.pad(mask, (0, padding), value=False)
    return context, mask


@register_architecture(
    "dual_system_self_attn",
    status="supported",
    note="DualSystem joint self-attention: MoT-style mixed attention at every layer.",
    framework="dual_system",
    variant="joint_self_attn",
)
class DualSystemSelfAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with true joint self-attention (FastWAM MoT pattern)."""

    def __init__(self, cfg=None):
        # self.video_backbone 是通过父类初始化的，会在这里加载 encoder (e.g. V-JEPA) 并完成对应 DiT 的初始化（包括对 encoder 输入输出的兼容，修改 patch_embedding 和 output_head）
        # LINK: openwam/model/architectures/base.py:189
        super().__init__(cfg)
        self._mot_driver: DualSystemMoTDriver | None = None
        self._mot_driver_kwargs: dict = {}
        self._compiled_mot_run_joint_loop: Optional[Callable[..., Tuple[object, object]]] = None
        self.action_router = None
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096, Cosmos-Predict2.5=1024); explicit cfg/CLI
        # still wins. Falls back to 4096 when the backbone doesn't expose it.
        _vb_text_dim = getattr(self.video_backbone, "text_dim", None)
        text_dim = int(self._cfg_get(cfg, "text_dim", _vb_text_dim or 4096))
        self._init_proprio_context(cfg, text_dim=text_dim)

        # FastWAM-Joint compat: action residual hidden_dim may differ from
        # video_dim. The MoT driver only requires num_heads / attn_head_dim
        # parity (validated at DualSystemMoTDriver.__init__).
        action_dim_hidden = int(cfg.get("dim", 1024))
        num_heads = int(cfg.get("num_heads", 24))
        attn_head_dim = int(cfg.get("attn_head_dim", video_dim // num_heads))

        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 4 * action_dim_hidden)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_self_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=cfg.get("shift_action"),
        )

        # MoT driver is built once both backbones are available. The video
        # backbone is normally constructed in ``BaseWAMArchitecture._init_video_backbone``
        # (already done by ``super().__init__``), so we can build the driver
        # right here. Tests that swap in a mock video backbone after init call
        # :meth:`build_mot_driver` directly.
        self._mot_driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(cfg.get("mot_checkpoint_mixed_attn", True)),
            "attention_mask_mode": str(cfg.get("attention_mask_mode", ACTION_SEES_VIDEO)),
            "video_attention_mask_mode": str(cfg.get("video_attention_mask_mode", "clean_condition_causal")),
        }
        if self.video_backbone is not None:
            self.build_mot_driver()

        router_cfg = cfg.get("action_router", {})
        enabled = router_cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("action_router.enabled must be a boolean")
        if enabled:
            if self._mot_driver_kwargs["video_attention_mask_mode"] != "clean_condition_causal":
                raise ValueError("action_router requires clean_condition_causal visual isolation")
            # Adding a router must not consume the baseline initialization RNG.
            with torch.random.fork_rng(devices=[]):
                self.action_router = ActionVisualRouter(
                    action_dim_hidden, video_dim,
                    **{k: v for k, v in router_cfg.items() if k != "enabled"},
                )

    def build_mot_driver(self) -> DualSystemMoTDriver:
        """Construct the :class:`DualSystemMoTDriver` from the current backbones.

        Re-callable; raises if either backbone is missing. Tests that swap in
        a mock video backbone after ``__init__`` should call this method to
        wire up the driver afterwards.
        """
        if self.video_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: video_backbone is not "
                "set. Construct the architecture with a video_backbone config or attach "
                "one before calling this method."
            )
        if self.action_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: action_backbone is not "
                "set. Architecture must be built from a non-None cfg."
            )

        self._mot_driver = DualSystemMoTDriver(
            self.video_backbone,
            self.action_backbone,
            **self._mot_driver_kwargs,
        )
        return self._mot_driver

    @property
    def mot_driver(self) -> DualSystemMoTDriver | None:
        """The MoT joint-attention driver (None if the architecture wasn't fully built)."""
        return self._mot_driver

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile the eval-time MoT joint loop when requested."""

        super().apply_compile_optimizations(compile_cfg)
        self._compiled_mot_run_joint_loop = None
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = self_attn_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("dual self-attn MoT compile disabled by config; running eager.")
            return
        if self.video_backbone is None or self.action_backbone is None:
            logger.warning("dual self-attn MoT compile requested before backbones are ready; running eager.")
            return

        driver = self._mot_driver or self.build_mot_driver()

        def _run_joint_loop(vstate, astate):
            return driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

        kwargs = torch_compile_kwargs(section, default_mode="reduce-overhead")
        try:
            self._compiled_mot_run_joint_loop = torch.compile(_run_joint_loop, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive setup fallback
            self._compiled_mot_run_joint_loop = None
            logger.warning("dual self-attn MoT torch.compile setup failed; running eager: %s", exc)
            return
        logger.info("Enabled dual self-attn MoT compile with torch.compile kwargs=%s", kwargs)

    @timed("joint_forward")
    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        return_router_aux: bool = False,
        router_action_is_pad: Optional[Tensor] = None,
        use_action_router: bool = True,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]] | Tuple[Tensor, Optional[Tensor], Tensor]:
        vb = self.video_backbone        # Wan22Ti2v 类型
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )

        from openwam.model.video_backbone.wan_backbone import WanBase

        # I2V splits image/text tokens at a fixed 512-token boundary.
        if isinstance(vb, WanBase) and not vb.dit.has_image_input and pipeline_inputs.get("context") is not None:
            pipeline_inputs["context"], pipeline_inputs["context_mask"] = _fixed_wan_text_context(
                pipeline_inputs["context"], pipeline_inputs.get("context_mask"), pipeline_inputs.get("seq_lens")
            )
            if pipeline_inputs.get("seq_lens") is not None:
                pipeline_inputs["seq_lens"] = pipeline_inputs["seq_lens"].clamp(max=128)
        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        repa_layer = pipeline_inputs.pop("repa_layer", None)
        if repa_layer is not None and not 0 <= repa_layer < vb.num_layers:
            raise ValueError(f"repa_layer must be in [0, {vb.num_layers})")
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Opt every Wan backbone into 4D + clean-prefix-aligned t_mod (mirrors
        # TI2V's native ``seperated_timestep + fuse_vae_embedding_in_latents``
        # path; TI2V itself fires that path first so these kwargs are inert for
        # it). VACE and I2V do NOT emit ``clean_condition_latents`` (VACE routes
        # its first-frame condition through ``vace_context``; I2V uses the
        # ``y`` channel), so for them ``zero_clean_prefix_t_mod`` is
        # structurally inert — kept on only so the joint MoT driver gets the
        # 4D ``t_mod`` it needs. ``setdefault`` so explicit callers can still
        # pass False.
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        # LINK: openwam/model/video_backbone/wan_backbone.py:350
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        if noisy_actions is None or ab is None:
            repa_hidden = None
            for block_id in range(vb.num_layers):
                vstate = vb.run_block(block_id, vstate)
                if block_id == repa_layer:
                    repa_hidden = vstate.hidden_states
            if repa_layer is not None:
                return vb.finalize(vstate), None, repa_hidden
            return vb.finalize(vstate), None

        if isinstance(vb, WanBase):
            vb.compact_time_modulation_for_joint(vstate)

        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()

        astate = ab.prepare_state(
            noisy_actions,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        compiled_loop = getattr(self, "_compiled_mot_run_joint_loop", None)
        if (compiled_loop is not None and repa_layer is None
                and not use_gradient_checkpointing and not use_gradient_checkpointing_offload):
            try:
                vstate, astate = compiled_loop(vstate, astate)
            except Exception as exc:
                self._compiled_mot_run_joint_loop = None
                logger.warning("dual self-attn compiled MoT loop failed; falling back to eager: %s", exc)
                vstate, astate = driver.run_joint_loop(
                    vstate,
                    astate,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
        else:
            loop_result = driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                capture_layer=repa_layer,
            )
            if repa_layer is not None:
                vstate, astate, repa_hidden = loop_result
            else:
                vstate, astate = loop_result
        router_aux = {}
        if self.action_router is not None and use_action_router:
            before = astate.payload.x_action
            valid = None
            if router_action_is_pad is not None:
                pad = router_action_is_pad.bool()
                valid = ~pad.all(-1) if pad.ndim == 3 else ~pad
            clean_count = int(vstate.extras["num_clean_condition_latents"]) * int(vstate.grid_height) * int(vstate.grid_width)
            clean = vstate.hidden_states[:, :clean_count]
            view_ids = None
            if getattr(vb, "_seperate_encode_view", False):
                width = int(vstate.grid_width)
                view_ids = (torch.arange(clean_count, device=clean.device) % width) // (width // 2)
            updated, router_aux = self.action_router(
                before, clean, query_valid=valid, view_ids=view_ids, return_aux=return_router_aux,
            )
            astate.payload.x_action = updated
            if return_router_aux:
                with torch.no_grad():
                    change = torch.nn.functional.linear(
                        (updated - before).detach(), ab.action_decoder.weight.detach()).float()
                    cells = torch.ones_like(change, dtype=torch.bool)
                    if router_action_is_pad is not None:
                        pad = router_action_is_pad.to(change.device).bool()
                        cells = (~pad).unsqueeze(-1).expand_as(change) if pad.ndim == 2 else ~pad
                    router_aux["router_action_delta_rms"] = (
                        (change.square() * cells).sum() / cells.sum().clamp_min(1)).sqrt()
        result = (vb.finalize(vstate), ab.extract_prediction(astate))
        if repa_layer is not None:
            result += (repa_hidden,)
        if return_router_aux:
            result += (router_aux,)
        return result


__all__ = ["DualSystemSelfAttnArchitecture"]
