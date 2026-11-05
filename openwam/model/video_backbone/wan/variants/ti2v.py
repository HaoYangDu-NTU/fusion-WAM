"""TI2V variant: the clean condition prefix is kept noise-free."""

from __future__ import annotations

from openwam.model.video_backbone.wan.variants.base import WanVariant


class TI2VVariant(WanVariant):
    """Pin t=0 on the clean condition tokens and exclude them from video loss.

    Native VAE / H=0 uses one latent; V-JEPA history uses H latents.
    Conditioning is enabled only when reference images are provided.
    """

    needs_clean_condition_skip = True

    def build_train_conditioning(self, bb, *, input_latents, ref_images, **kw) -> dict:
        # Reuse the independently encoded clean prefix from the encoder output.
        has_ref = ref_images is not None and ref_images[0] is not None
        if not has_ref:
            return {}
        encoder = getattr(bb, "video_encoder", None)
        history = getattr(encoder, "history_num_tubelets", None)
        n = max(1, int(history)) if history is not None else int(
            getattr(getattr(encoder, "properties", None), "num_condition_latents", 1)
        )
        return {
            # Learned fusion is deferred past no_grad preprocessing; the loss
            # entry fills this slice from the live fused representation.
            "clean_condition_latents": input_latents[:, :, :n].clone() if input_latents is not None else None,
            "num_clean_condition_latents": n,
            "fuse_vae_embedding_in_latents": True,
        }
