"""Action-only ablation that retains the masked future-token branch."""

from vjepa_policy.experimental.action_only_policy import (
    ActionOnlyVJEPAPolicy,
    build_action_only_policy,
)


class FutureBranchActionOnlyVJEPAPolicy(ActionOnlyVJEPAPolicy):
    """Run masked future attention without future-label supervision."""

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
        del return_prediction
        _, predictor_kv = self.predictor(
            context_latents,
            mask_enc,
            mask_pred,
            language=language,
            language_mask=language_mask,
            proprio=proprio,
            # The future tokens and all bidirectional blocks still execute;
            # the un-supervised latent projection is unnecessary for action loss.
            return_prediction=False,
        )
        return predictor_kv


def build_future_branch_action_only_policy(encoder, predictor, **kwargs):
    """Build the full future-branch policy with action loss only."""
    return build_action_only_policy(
        encoder,
        predictor,
        policy_class=FutureBranchActionOnlyVJEPAPolicy,
        **kwargs,
    )
