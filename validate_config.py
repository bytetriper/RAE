#!/usr/bin/env python3
"""
validate_config.py
Zero-dependency config validator for RAE + DiT
Author: GitHub@YOUR_USERNAME
Usage: python validate_config.py --config path/to/config.yaml
"""

import argparse
import yaml
import sys
import os

REQUIRED_TOP_KEYS = {
    "stage_1", "stage_2", "transport", "sampler", "guidance", "misc", "training", "eval"
}

REQUIRED_STAGE1_KEYS = {"target", "params"}
REQUIRED_STAGE2_KEYS = {"target", "params"}
REQUIRED_MISC_KEYS = {"latent_size", "num_classes"}

def validate_config(path):
    if not os.path.isfile(path):
        return False, f"File not found: {path}"
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, f"Invalid YAML: {e}"

    if not isinstance(cfg, dict):
        return False, "Top-level must be a dict"

    missing = REQUIRED_TOP_KEYS - cfg.keys()
    if missing:
        return False, f"Missing top-level keys: {sorted(missing)}"

    if not isinstance(cfg.get("stage_1"), dict):
        return False, "stage_1 must be a dict"
    missing = REQUIRED_STAGE1_KEYS - cfg["stage_1"].keys()
    if missing:
        return False, f"stage_1 missing: {sorted(missing)}"

    if not isinstance(cfg.get("stage_2"), dict):
        return False, "stage_2 must be a dict"
    missing = REQUIRED_STAGE2_KEYS - cfg["stage_2"].keys()
    if missing:
        return False, f"stage_2 missing: {sorted(missing)}"

    if not isinstance(cfg.get("misc"), dict):
        return False, "misc must be a dict"
    missing = REQUIRED_MISC_KEYS - cfg["misc"].keys()
    if missing:
        return False, f"misc missing: {sorted(missing)}"

    latent = cfg["misc"]["latent_size"]
    if not (isinstance(latent, list) and len(latent) == 3 and all(isinstance(x, int) for x in latent)):
        return False, "misc.latent_size must be list of 3 ints"

    if not isinstance(cfg["misc"]["num_classes"], int) or cfg["misc"]["num_classes"] <= 0:
        return False, "misc.num_classes must be positive int"

    return True, "Config is valid"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate RAE config YAML")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    ok, msg = validate_config(args.config)
    print(msg)
    sys.exit(0 if ok else 1)
