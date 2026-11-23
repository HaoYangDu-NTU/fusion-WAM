"""Cache frozen E05 joint-stereo features once per actual LIBERO timestamp."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openwam.model.video_backbone.encoder.vggt_omega import VGGTOmegaSpatialEncoder
from scripts.feature_cache.spatial import build_cache


def main():
    build_cache("E05", VGGTOmegaSpatialEncoder, __doc__)


if __name__ == "__main__":
    main()
