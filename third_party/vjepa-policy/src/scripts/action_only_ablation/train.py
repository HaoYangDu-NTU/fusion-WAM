"""Train the isolated action-only V-JEPA policy ablation."""

from scripts.train_vjepa_policy import main
from vjepa_policy.experimental.action_only_policy import build_action_only_policy


if __name__ == "__main__":
    main(policy_builder=build_action_only_policy)
