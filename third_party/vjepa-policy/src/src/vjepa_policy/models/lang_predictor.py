# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Derived from src/models/predictor.py (VisionTransformerPredictor): the rollout
# backbone is unchanged; a zero-init FiLM branch conditions it on a language
# embedding (e.g. T5 encoder hidden states).

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_policy.models.backbone.masks import apply_masks
from vjepa_policy.models.backbone.modules import Block
from vjepa_policy.models.backbone.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from vjepa_policy.models.backbone.tensors import repeat_interleave_batch, trunc_normal_


def build_context_future_attention_mask(is_context):
    """Build clean visibility: context reads context; future reads all tokens."""
    if is_context.ndim not in (1, 2):
        raise ValueError(f"is_context must be [N] or [B,N], got {tuple(is_context.shape)}")
    is_context = is_context.bool()
    allow = (~is_context.unsqueeze(-1)) | is_context.unsqueeze(-2)
    return allow if is_context.ndim == 1 else allow.unsqueeze(1)


class LangConditionedVisionTransformerPredictor(nn.Module):
    """V-JEPA predictor with zero-init FiLM language conditioning.

    Identical to ``VisionTransformerPredictor`` except that, before every
    transformer block, the token stream is modulated by a per-layer FiLM
    transform::

        x <- x * (1 + gamma_l) + beta_l

    where ``(gamma_l, beta_l)`` are produced from a language embedding
    (per-video T5 encoder hidden states, batch preserved). The FiLM heads are
    zero-initialised, so at init ``gamma_l = beta_l = 0`` and this module is
    *exactly* the original predictor -- language influence grows from zero.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=False,
        lang_dim=768,        # T5 hidden size: t5-base=768, t5-large/3b/11b=1024, flan-t5-xxl=4096
        film_hidden=None,    # FiLM trunk width; defaults to predictor_embed_dim
        use_clean_context_attention=False,
        **kwargs
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.chop_last_n_tokens = chop_last_n_tokens

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for i in range(num_mask_tokens)]
            )

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing
        self.use_clean_context_attention = use_clean_context_attention

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
            )
        else:
            self.num_patches = num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        # Position embedding
        self.uniform_power = uniform_power

        self.predictor_pos_embed = None
        if not use_rope:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
            )

        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if out_embed_dim is None:
            teacher_embed_dim = kwargs.get("teacher_embed_dim", None)
            if teacher_embed_dim is not None:
                out_embed_dim = teacher_embed_dim
            else:
                out_embed_dim = embed_dim

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, out_embed_dim, bias=True)

        # ------ zero-init FiLM language conditioning
        # Pool the T5 sequence to one vector per video, then produce a per-layer
        # (gamma, beta) pair. The final head of every layer is zero-initialised
        # so the block loop starts as an identity w.r.t. language.
        if film_hidden is None:
            film_hidden = predictor_embed_dim
        self.lang_dim = lang_dim
        self.lang_norm = nn.LayerNorm(lang_dim)
        self.lang_trunk = nn.Sequential(
            nn.Linear(lang_dim, film_hidden, bias=True),
            nn.GELU(),
        )
        self.film_heads = nn.ModuleList(
            [nn.Linear(film_hidden, 2 * predictor_embed_dim, bias=True) for _ in range(depth)]
        )

        # ------ initialize weights
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        if self.mask_tokens is not None and not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()
        
        # Zero the FiLM heads *after* generic init so conditioning starts off.
        for head in self.film_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size  # TODO: update; currently assumes square input
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _film_params(self, lang, lang_mask):
        """Pool T5 hidden states over the sequence and build per-layer (gamma, beta).

        :param lang: language hidden states [B, S, lang_dim] (e.g. T5 last_hidden_state)
        :param lang_mask: optional token mask [B, S] (T5 attention_mask); 1 = keep
        :return: list of ``depth`` tuples (gamma, beta), each [B, predictor_embed_dim]
        """
        if lang_mask is None:
            pooled = lang.mean(dim=1)
        else:
            w = lang_mask.unsqueeze(-1).to(lang.dtype)  # [B, S, 1]
            pooled = (lang * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
        g = self.lang_trunk(self.lang_norm(pooled))  # [B, film_hidden]
        return [head(g).chunk(2, dim=-1) for head in self.film_heads]

    def forward(
        self, x, masks_x, masks_y, mask_index=1, has_cls=False, lang=None, lang_mask=None, capture_kv=None,
    ):
        """
        :param x: context tokens
        :param masks_x: indices of context tokens in input
        :params masks_y: indices of target tokens in input
        :param lang: language hidden states [B, S, lang_dim]; None disables FiLM
        :param lang_mask: optional language token mask [B, S]
        :param capture_kv: optional list; if given, each block appends its
            post-RoPE (k, v) to it in layer order, so ``capture_kv[i]`` is
            layer ``i``'s (k, v) after the call. Read-only side channel for an
            external module (e.g. an action expert) to attend into this
            predictor's own keys/values -- does not affect this forward's
            output. Not supported together with activation checkpointing.
        """
        if capture_kv is not None and self.use_activation_checkpointing:
            raise NotImplementedError(
                "capture_kv is not supported with use_activation_checkpointing=True "
                "(checkpoint recompute would double-append to the list)."
            )
        assert (masks_x is not None) and (masks_y is not None), "Cannot run predictor without mask indices"
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        # Batch Size
        B = len(x) // len(masks_x)

        # Map context tokens to predictor dimensions
        x = self.predictor_embed(x)
        if has_cls:
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]
        _, N_ctxt, D = x.shape

        # Add positional embedding to ctxt tokens
        if not self.use_rope:
            x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embed, masks_x)

        # Make target tokens
        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)
        # -- add pos embed
        if not self.use_rope:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_y)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens += pos_embs

        # Concatenate context & target tokens
        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Positions of context & target tokens
        masks_x = torch.cat(masks_x, dim=0)
        masks_y = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x, masks_y], dim=1)
        is_context = None
        if self.use_clean_context_attention:
            is_context = torch.cat(
                [torch.ones_like(masks_x, dtype=torch.bool), torch.zeros_like(masks_y, dtype=torch.bool)],
                dim=1,
            )

        # Put tokens in sorted order
        argsort = torch.argsort(masks, dim=1)  # [B, N]
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)
        if is_context is not None:
            is_context = torch.stack([is_context[i, row] for i, row in enumerate(argsort)], dim=0)

        # Remove the last n tokens of sorted sequence before processing
        if self.chop_last_n_tokens > 0:
            x = x[:, : -self.chop_last_n_tokens]
            masks = masks[:, : -self.chop_last_n_tokens]
            if is_context is not None:
                is_context = is_context[:, : -self.chop_last_n_tokens]

        if has_cls:
            x = torch.cat([x_cls, x], dim=1)
            if is_context is not None:
                is_context = torch.cat(
                    [torch.ones((is_context.shape[0], 1), dtype=torch.bool, device=x.device), is_context],
                    dim=1,
                )

        attn_mask = None
        if is_context is not None:
            if is_context.shape[0] == 1 or torch.equal(is_context, is_context[:1].expand_as(is_context)):
                attn_mask = build_context_future_attention_mask(is_context[0])
            else:
                attn_mask = build_context_future_attention_mask(is_context)

        # Per-layer FiLM params from language (batch preserved). Zero at init.
        film = self._film_params(lang, lang_mask) if lang is not None else None

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if film is not None:
                gamma, beta = film[i]
                # x was tiled by len(masks_x) context views; match the FiLM batch.
                if gamma.size(0) != x.size(0):
                    r = x.size(0) // gamma.size(0)
                    gamma = gamma.repeat(r, 1)
                    beta = beta.repeat(r, 1)
                x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    masks,
                    attn_mask,
                    T=self.grid_depth,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=masks,
                    attn_mask=attn_mask,
                    T=self.grid_depth,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    kv_out=capture_kv,
                )
        x = self.predictor_norm(x)

        if has_cls:
            x = x[:, 1:, :]

        # Return output corresponding to target tokens
        if not self.return_all_tokens:
            reverse_argsort = torch.argsort(argsort, dim=1)  # [B, N]
            x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
            x = x[:, N_ctxt:]

        x = self.predictor_proj(x)

        return x


def lang_vit_predictor(**kwargs):
    model = LangConditionedVisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


# --------------------------------------------------------------------------- #
# World-model training wrapper (NO action expert).
#
# This is the "language-conditioned V-JEPA world model" architecture, kept fully
# self-contained here and INDEPENDENT of the action policy (see
# vjepa_policy.models.vjepa_policy.VJEPAPolicy). The two share the frozen encoder
# and this predictor by composition, but neither subclasses the other, so the two
# architectures can evolve without coupling.
# --------------------------------------------------------------------------- #
class LangJEPA(nn.Module):
    """Language-conditioned latent world model: frozen encoder + FiLM predictor.

    The encoder is frozen and serves as *both* the context encoder and the target
    (they are identical), so there is no EMA. Only the predictor (incl. its FiLM
    language branch) is trained: given the context tokens (first tubelet) and the
    task language, it predicts the frozen-encoder latents of the future tokens.
    ``forward`` returns the JEPA L1 loss so DDP grad sync happens on backward.
    """

    def __init__(self, encoder, predictor, loss_exp=1.0, patch_valid=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_exp = loss_exp
        if patch_valid is not None:
            self.register_buffer("patch_valid", patch_valid.bool(), persistent=False)
        else:
            self.patch_valid = None
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, batch):
        clips, lang, lang_mask, masks_enc, masks_pred = batch

        # Stash per-layer FiLM modulation magnitude for logging (cheap, off-graph):
        # gamma/beta are the (1+gamma)*x + beta modulation; |.| ~ how far from identity.
        with torch.no_grad():
            film = self.predictor._film_params(lang, lang_mask)  # list[(gamma, beta)], each [B, Dp]
            self.film_gamma_abs = torch.stack([g.abs().mean() for g, _ in film]).detach()  # [depth]
            self.film_beta_abs = torch.stack([b.abs().mean() for _, b in film]).detach()   # [depth]

        # Encoder is frozen -> both passes run without grad (saves memory).
        with torch.no_grad():
            h = self.encoder(clips)  # full-clip target latents [B, N, D]
            h = F.layer_norm(h, (h.size(-1),))

        loss = clips.new_zeros(())
        cos = clips.new_zeros(())
        for m_enc, m_pred in zip(masks_enc, masks_pred):
            with torch.no_grad():
                z_ctx = self.encoder(clips, m_enc)  # context tokens [B, Kc, D]
            # mask_index defaults to 1 in the predictor -> matches zero-shot V-JEPA2.
            z = self.predictor(z_ctx, m_enc, m_pred, lang=lang, lang_mask=lang_mask)
            target = apply_masks(h, [m_pred])  # [B, Kp, D]
            per_token = torch.mean(torch.abs(z - target) ** self.loss_exp, dim=-1) / self.loss_exp
            cos_per = F.cosine_similarity(z.float(), target.float(), dim=-1)
            if self.patch_valid is not None:
                valid_t = self.patch_valid[m_pred].to(per_token.dtype)
                denom = valid_t.sum().clamp_min(1.0)
                loss = loss + (per_token * valid_t).sum() / denom
                cos = cos + (cos_per * valid_t).sum() / denom
            else:
                loss = loss + per_token.mean()
                cos = cos + cos_per.mean()
        n = len(masks_enc)
        loss = loss / n
        return loss, {"loss": loss.detach(), "cos_sim": (cos / n).detach()}
