"""Pure-context action-only V-JEPA policy ablation."""

from vjepa_policy.experimental.action_only_policy import (
    ActionOnlyVJEPAPolicy,
    build_action_only_policy,
)


class ContextOnlyVJEPAPolicy(ActionOnlyVJEPAPolicy):
    """Train and serve actions without instantiating future mask tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_only = True
        # Keep the canonical state-dict topology for serving, but do not train
        # the unused future-token parameters in this ablation.
        for token in self.predictor.mask_tokens:
            token.requires_grad_(False)

    def _predictor_forward(
        self,
        context_latents,
        mask_enc,
        mask_pred,
        *,
        language,
        language_mask,
        proprio,
        return_prediction,
    ):
        del mask_pred, return_prediction
        return self.predictor.forward_context_only(
            context_latents,
            mask_enc,
            language=language,
            language_mask=language_mask,
            proprio=proprio,
        )


def build_context_only_policy(encoder, predictor, **kwargs):
    """Build the canonical action topology with a pure-context Predictor path."""
    return build_action_only_policy(
        encoder,
        predictor,
        policy_class=ContextOnlyVJEPAPolicy,
        **kwargs,
    )
