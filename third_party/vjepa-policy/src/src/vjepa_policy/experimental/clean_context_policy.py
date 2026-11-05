"""Policy builder for the Route 2 clean-context/full-KV ablation."""

from vjepa_policy.experimental.route2_policy import build_route2_baseline_policy


def build_clean_context_policy(*args, **kwargs):
    policy = build_route2_baseline_policy(*args, **kwargs)
    policy.action_expert.read_full_predictor_kv = True
    return policy
