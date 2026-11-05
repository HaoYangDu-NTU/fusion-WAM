#!/usr/bin/env python3
"""Read-only asset checks; no torch import, GPU allocation, or training."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha256", action="store_true", help="Hash large weight files too")
    parser.add_argument("--validate-caches", action="store_true", help="Read and validate every cached tensor")
    parser.add_argument("--require-ready", action="store_true", help="Require assets, validated caches and GPU acceptance")
    args = parser.parse_args()
    registry = json.loads((ROOT / "assets/registry.json").read_text())
    report = {
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "check_level": "sha256" if args.sha256 else "existence_size_config",
        "pretrained": {},
        "blockers": [],
    }
    assets_ok = True
    for name, asset in registry["pretrained"].items():
        directory = Path(asset["directory"])
        entry = {"directory": str(directory), "files": {}, "ok": True}
        for filename, expected_size in asset["required_files"].items():
            path = directory / filename
            actual_size = path.stat().st_size if path.is_file() else None
            item = {"exists": path.is_file(), "bytes": actual_size,
                    "expected_bytes": expected_size, "ok": actual_size == expected_size}
            if args.sha256 and item["exists"]:
                item["sha256"] = sha256(path)
                reference = asset.get("reference_sha256", {}).get(filename)
                if reference:
                    item["reference_sha256_matches"] = item["sha256"] == reference
                    item["ok"] = item["ok"] and item["reference_sha256_matches"]
            entry["files"][filename] = item
            entry["ok"] = entry["ok"] and item["ok"]
        config_path = directory / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            geometry = config.get("text_config", config)
            entry["hidden_size"] = geometry.get(asset.get("hidden_size_config_key", "hidden_size"))
            entry["geometry_matches"] = entry["hidden_size"] == asset["hidden_size"]
            entry["ok"] = entry["ok"] and entry["geometry_matches"]
        link = ROOT / "assets/pretrained" / asset["link_name"]
        entry["asset_link_matches"] = link.is_symlink() and link.resolve() == directory.resolve()
        entry["ok"] = entry["ok"] and entry["asset_link_matches"]
        assets_ok = assets_ok and entry["ok"]
        report["pretrained"][name] = entry
        print(f"{name}: {'OK' if entry['ok'] else 'FAIL'}")

    expected_caches = {
        "libero": ROOT / "cache/text/rynnbrain1p1-9b/libero/avg_all_layers/manifest.json",
        "libero_plus": ROOT / "cache/text/rynnbrain1p1-9b/libero-plus/avg_all_layers/corrected_text/manifest.json",
    }
    report["cache_manifests_exist"] = {key: path.is_file() for key, path in expected_caches.items()}
    for key, exists in report["cache_manifests_exist"].items():
        if not exists:
            report["blockers"].append(f"{key}: RynnBrain 9B text cache not generated")
    report["backend_entry_exists"] = all(
        (ROOT / name).is_file() for name in (
            "scripts/train.py", "scripts/experiment.py", "openwam/experiments.py",
            "configs/experiments/E00.yaml", "configs/experiments/E01.yaml", "configs/experiments/E02.yaml",
        )
    )
    if not report["backend_entry_exists"]:
        report["blockers"].append("Native experiment entrypoints missing")
    report["cache_validation"] = {}
    if args.validate_caches or args.require_ready:
        import sys
        sys.path.insert(0, str(ROOT))
        from openwam.experiments import LIBERO_CATALOG, PLUS_CATALOG, validate_cache
        from openwam.text_condition.precompute import model_fingerprint
        fingerprint = model_fingerprint(registry["pretrained"]["rynnbrain_9b"]["directory"])
        for key, path in expected_caches.items():
            if path.is_file():
                try:
                    catalog = LIBERO_CATALOG if key == "libero" else PLUS_CATALOG / "corrected_text/manifest.json"
                    report["cache_validation"][key] = {"ok": True, **validate_cache(path.parent, catalog, fingerprint)}
                except (ValueError, KeyError, OSError) as exc:
                    report["cache_validation"][key] = {"ok": False, "error": str(exc)}
                    report["blockers"].append(f"{key}: {exc}")
    report["assets_ok"] = assets_ok
    report["training_inputs_ready"] = (
        assets_ok and report["backend_entry_exists"]
        and report["cache_validation"].get("libero", {}).get("ok", False)
    )
    # GPU acceptance is a reviewed measurement, never inferred from file existence.
    report["gpu_smoke_status"] = "pending"
    report["training_ready"] = False
    report["blockers"].append("GPU smoke, throughput and four-rank resume acceptance pending")
    destination = ROOT / "reports/asset_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {destination}")
    for blocker in report["blockers"]:
        print(f"PENDING: {blocker}")
    cache_failed = any(not result["ok"] for result in report["cache_validation"].values())
    return 1 if not assets_ok or cache_failed or (args.require_ready and not report["training_ready"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
