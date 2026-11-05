"""Move legacy UMT5 caches into encoder/dataset directories, retaining old aliases.

Tensor files are unchanged: legacy 512-token caches are cropped on read to 128
for new configs, preserving the original UMT5 feature values and old runs.
"""
import argparse
from pathlib import Path


def migrate(root):
    root = Path(root).resolve()
    for old, new in ((root / "text_embedding", root / "umt5-embedding/libero"),
                     (root / "libero-plus-embedding", root / "umt5-embedding/libero-plus")):
        if old.is_symlink():
            if old.resolve() != new.resolve():
                raise ValueError(f"Unexpected existing cache alias: {old}")
            continue
        if new.exists():
            raise FileExistsError(f"Refusing to overwrite {new}")
        if not old.is_dir():
            raise FileNotFoundError(old)
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        old.symlink_to(new.relative_to(root), target_is_directory=True)
        print(f"{new} (legacy alias: {old})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    migrate(parser.parse_args().data_root)
