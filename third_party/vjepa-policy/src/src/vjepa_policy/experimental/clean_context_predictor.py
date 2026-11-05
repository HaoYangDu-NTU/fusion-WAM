"""Route 2 Predictor with clean-context attention and full per-layer K/V."""

from functools import partial

import torch

from vjepa_policy.models.backbone.masks import apply_masks
from vjepa_policy.models.lang_predictor import build_context_future_attention_mask
from vjepa_policy.models.wm_predictor import WorldModelPredictor


class CleanContextWorldModelPredictor(WorldModelPredictor):
    """Restrict context queries to context while future queries see all tokens."""

    use_clean_context_attention = True

    @staticmethod
    def _forward_clean_block(
        block,
        tokens,
        condition,
        condition_valid,
        position_ids,
        attention_mask,
        *,
        T,
        H_patches,
        W_patches,
    ):
        attention_output, key, value = block.self_attn(
            block.self_norm(tokens),
            mask=position_ids,
            attn_mask=attention_mask,
            T=T,
            H_patches=H_patches,
            W_patches=W_patches,
            return_kv=True,
        )
        tokens = tokens + block.drop_path(attention_output)
        tokens = tokens + block.drop_path(
            block.cross_attn(block.cross_norm(tokens), condition, condition_valid)
        )
        tokens = tokens + block.drop_path(block.ffn(block.ffn_norm(tokens)))
        return tokens, key, value

    def forward(
        self,
        context,
        masks_x,
        masks_y,
        *,
        language,
        language_mask,
        proprio,
        mask_index=1,
    ):
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]
        if len(masks_x) != len(masks_y):
            raise ValueError("masks_x and masks_y must have the same length")
        if not masks_x:
            raise ValueError("at least one context/future mask pair is required")

        condition, condition_valid = self._condition_tokens(
            language, language_mask, proprio
        )
        context = self.predictor_embed(context)
        batch_size = context.shape[0] // len(masks_x)
        context_length = context.shape[1]
        mask_index %= self.num_mask_tokens
        future = self.mask_tokens[mask_index].expand(batch_size, self.num_patches, -1)
        future = apply_masks(future, masks_y)
        context = context.repeat(len(masks_x), 1, 1)
        tokens = torch.cat((context, future), dim=1)

        masks_x = torch.cat(masks_x, dim=0)
        masks_y = torch.cat(masks_y, dim=0)
        position_ids = torch.cat((masks_x, masks_y), dim=1)
        is_context = torch.cat(
            (
                torch.ones_like(masks_x, dtype=torch.bool),
                torch.zeros_like(masks_y, dtype=torch.bool),
            ),
            dim=1,
        )
        argsort = torch.argsort(position_ids, dim=1)
        position_ids = torch.gather(position_ids, 1, argsort)
        tokens = torch.gather(
            tokens,
            1,
            argsort.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        is_context = torch.gather(is_context, 1, argsort)
        if is_context.shape[0] == 1 or torch.equal(
            is_context, is_context[:1].expand_as(is_context)
        ):
            attention_mask = build_context_future_attention_mask(is_context[0])
        else:
            attention_mask = build_context_future_attention_mask(is_context)

        condition = self._match_batch(condition, tokens.shape[0])
        condition_valid = self._match_batch(condition_valid, tokens.shape[0])
        predictor_kv = []
        for block_index, block in enumerate(self.predictor_blocks):
            block_forward = partial(self._forward_clean_block, block)
            if block_index < self.activation_checkpointing_blocks and self.training:
                tokens, key, value = torch.utils.checkpoint.checkpoint(
                    block_forward,
                    tokens,
                    condition,
                    condition_valid,
                    position_ids,
                    attention_mask,
                    T=self.grid_depth,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    use_reentrant=False,
                )
            else:
                tokens, key, value = block_forward(
                    tokens,
                    condition,
                    condition_valid,
                    position_ids,
                    attention_mask,
                    T=self.grid_depth,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                )
            predictor_kv.append((key, value))

        tokens = self.predictor_norm(tokens)
        reverse_argsort = torch.argsort(argsort, dim=1)
        tokens = torch.gather(
            tokens,
            1,
            reverse_argsort.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        prediction = self.predictor_proj(tokens[:, context_length:])
        return prediction, tuple(predictor_kv)


def build_clean_context_predictor(**kwargs):
    return CleanContextWorldModelPredictor(**kwargs)
