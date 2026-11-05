"""Precompute frozen LIBERO text features; supports torchrun rank sharding."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.text_condition.precompute import add_encoder_arguments, write_caches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_encoder_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path)
    source.add_argument("--catalog-manifest", type=Path, help="Reuse exact prompt strings from an existing cache manifest")
    args = parser.parse_args()
    if args.catalog_manifest:
        prompts = list(json.loads(args.catalog_manifest.read_text())["prompts"])
    else:
        import pandas as pd
        prompts = [str(text).strip() for text in pd.read_parquet(args.dataset_dir / "meta/tasks.parquet").index]
    write_caches({"": prompts}, args, "libero")


if __name__ == "__main__":
    main()
