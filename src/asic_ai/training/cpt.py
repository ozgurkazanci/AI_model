"""Continued Pretraining (CPT) launcher.

Configures and launches CPT using Axolotl framework.
This is the LEAST critical training stage — can be skipped if base model
already has sufficient domain knowledge.

Design Doc Reference: Section 4, Stage 1
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "training" / "cpt_axolotl.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load CPT training configuration.

    Args:
        config_path: Path to YAML config. Uses default if None.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_prerequisites(config: dict[str, Any]) -> list[str]:
    """Check that all prerequisites for CPT are met.

    Returns:
        List of warning/error messages. Empty if all good.
    """
    issues: list[str] = []

    # Check datasets exist
    for ds in config.get("datasets", []):
        ds_path = Path(ds["path"])
        if not ds_path.exists():
            issues.append(f"Dataset not found: {ds_path}")

    # Check base model
    base_model = config.get("base_model", "")
    if not base_model:
        issues.append("No base_model specified in config")

    # Check output dir
    output_dir = Path(config.get("output_dir", "./outputs/cpt"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check general code mix ratio
    weights = [ds.get("weight", 1.0) for ds in config.get("datasets", [])]
    general_weight = sum(
        ds.get("weight", 1.0)
        for ds in config.get("datasets", [])
        if "general" in ds.get("path", "").lower()
    )
    total_weight = sum(weights)
    if total_weight > 0:
        general_ratio = general_weight / total_weight
        if general_ratio < 0.10:
            issues.append(
                f"General code mix is {general_ratio:.0%} — recommend ≥15% "
                "to prevent catastrophic forgetting"
            )

    return issues


def launch_axolotl(config_path: str | Path) -> int:
    """Launch Axolotl training.

    Args:
        config_path: Path to the Axolotl YAML config.

    Returns:
        Process return code.
    """
    cmd = [
        sys.executable, "-m", "axolotl.cli.train",
        str(config_path),
    ]

    logger.info("Launching CPT: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=False,
        check=False,
    )

    return result.returncode


def run_cpt(
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> int:
    """Main entry point for CPT.

    Args:
        config_path: Path to config. Uses default if None.
        dry_run: If True, validate only without training.

    Returns:
        0 on success, non-zero on failure.
    """
    config = load_config(config_path)
    issues = validate_prerequisites(config)

    if issues:
        for issue in issues:
            logger.warning("CPT prerequisite issue: %s", issue)

    if dry_run:
        logger.info("Dry run complete. Issues found: %d", len(issues))
        return 0 if not issues else 1

    actual_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return launch_axolotl(actual_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch CPT training")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Validate only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    sys.exit(run_cpt(args.config, args.dry_run))
