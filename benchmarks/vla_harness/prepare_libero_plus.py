"""Prepare Plus evaluation assets for either raw V-JEPA or S-VAE checkpoints."""

from benchmarks.vla_harness.prepare_libero_plus_svae import main, prepare_checkpoint

__all__ = ["prepare_checkpoint"]

if __name__ == "__main__":
    main()
