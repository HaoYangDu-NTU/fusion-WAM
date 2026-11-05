"""Train the future-branch, action-only V-JEPA policy ablation."""

from scripts.train_vjepa_policy import main
from vjepa_policy.experimental.future_branch_action_only_policy import (
    build_future_branch_action_only_policy,
)


if __name__ == "__main__":
    main(policy_builder=build_future_branch_action_only_policy)
