"""Cache frozen E06 independent image features once per actual LIBERO timestamp."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openwam.model.video_backbone.encoder.flux2_framewise import Flux2FramewiseEncoder
from scripts.feature_cache.spatial import build_cache


def main():
    build_cache("E06", Flux2FramewiseEncoder, __doc__)


if __name__ == "__main__":
    main()
