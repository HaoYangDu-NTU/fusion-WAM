"""Train Route 2 with clean-context attention and full Predictor K/V."""

from scripts.route2.baseline.train import run
from vjepa_policy.experimental.clean_context_policy import build_clean_context_policy
from vjepa_policy.experimental.clean_context_predictor import (
    build_clean_context_predictor,
)


if __name__ == "__main__":
    run(
        predictor_builder=build_clean_context_predictor,
        policy_builder=build_clean_context_policy,
    )
