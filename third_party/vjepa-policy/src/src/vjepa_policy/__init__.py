"""VJEPA-Policy: language-conditioned V-JEPA world model + flow-matching action expert.

Two decoupled architectures, managed separately:
  * models.lang_predictor.LangJEPA -- latent-similarity research model
  * models.vjepa_policy.VJEPAPolicy -- predictive flow policy
Backbone (encoder, predictor, attention utils) is vendored from the official
V-JEPA 2 code under models/ and models/backbone/.
"""

__version__ = "0.1.0"

from vjepa_policy.state import (
    DEFAULT_MAX_STATE_DIM,
    pack_normalized_state,
    packed_state_dim,
    prepare_proprio_state,
)

__all__ = [
    "__version__",
    "DEFAULT_MAX_STATE_DIM",
    "pack_normalized_state",
    "packed_state_dim",
    "prepare_proprio_state",
]
