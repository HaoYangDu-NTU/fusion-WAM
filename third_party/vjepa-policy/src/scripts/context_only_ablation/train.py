"""Train the pure-context V-JEPA policy ablation."""

from scripts.train_vjepa_policy import main
from vjepa_policy.experimental.context_only_policy import build_context_only_policy


if __name__ == "__main__":
    main(policy_builder=build_context_only_policy)
