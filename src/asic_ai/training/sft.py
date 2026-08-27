"""Supervised Fine-Tuning (SFT) launcher — Agent Trajectories.

Configures and launches SFT using Axolotl framework.
THIS IS THE MOST DISTINCTIVE PART OF THE PROJECT.

The model learns three things simultaneously:
1. Calling tools in correct format
2. Interpreting simulation output
3. What to do after failure ← MOST VALUABLE

Design Doc Reference: Section 4, Stage 2
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "training" / "sft_axolotl.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load SFT training configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_trajectories(data_path: str | Path) -> dict[str, Any]:
    """Validate SFT trajectory data before training.

    This is CRITICAL — format inconsistency is the #1 pitfall.

    Args:
        data_path: Path to trajectory JSONL file.

    Returns:
        Validation report.
    """
    path = Path(data_path)
    if not path.exists():
        return {"valid": False, "error": f"File not found: {path}"}

    total = 0
    valid = 0
    errors: list[dict[str, Any]] = []
    tool_call_formats: set[str] = set()

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total += 1
            try:
                data = json.loads(line.strip())

                # Check required fields
                if "conversations" not in data and "messages" not in data:
                    errors.append({
                        "line": line_num,
                        "error": "Missing 'conversations' or 'messages' field",
                    })
                    continue

                messages = data.get("conversations", data.get("messages", []))

                # Check role sequence
                prev_role = None
                for msg in messages:
                    role = msg.get("role", msg.get("from", ""))
                    if not role:
                        errors.append({
                            "line": line_num,
                            "error": "Message missing role",
                        })

                    # Track tool call format consistency
                    content = msg.get("content", msg.get("value", ""))
                    if "<tool_call>" in str(content):
                        # Extract format signature
                        fmt = _extract_tool_call_format(str(content))
                        tool_call_formats.add(fmt)

                    prev_role = role

                valid += 1

            except json.JSONDecodeError as e:
                errors.append({
                    "line": line_num,
                    "error": f"JSON parse error: {e}",
                })

    # Check format consistency
    format_consistent = len(tool_call_formats) <= 1
    if not format_consistent:
        logger.warning(
            "CRITICAL: Found %d different tool call formats! Must be exactly 1.",
            len(tool_call_formats),
        )

    return {
        "valid": len(errors) == 0 and format_consistent,
        "total": total,
        "valid_count": valid,
        "error_count": len(errors),
        "errors": errors[:20],  # First 20 errors
        "tool_call_formats": list(tool_call_formats),
        "format_consistent": format_consistent,
    }


def _extract_tool_call_format(content: str) -> str:
    """Extract the structural format of a tool call (ignoring values)."""
    import re
    # Normalize: replace values with placeholders
    normalized = re.sub(r'"[^"]*":\s*"[^"]*"', '"K": "V"', content)
    normalized = re.sub(r'"[^"]*":\s*[\d.]+', '"K": 0', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized[:200]  # First 200 chars as format signature


def validate_prerequisites(config: dict[str, Any]) -> list[str]:
    """Check SFT prerequisites."""
    issues: list[str] = []

    # Check datasets
    for ds in config.get("datasets", []):
        ds_path = Path(ds["path"])
        if not ds_path.exists():
            issues.append(f"Dataset not found: {ds_path}")
        else:
            report = validate_trajectories(ds_path)
            if not report["valid"]:
                issues.append(
                    f"Dataset validation failed: {ds_path} — "
                    f"{report['error_count']} errors, "
                    f"format consistent: {report['format_consistent']}"
                )

    # Check base model / CPT checkpoint
    base_model = config.get("base_model", "")
    if base_model.startswith("./") and not Path(base_model).exists():
        issues.append(
            f"CPT checkpoint not found: {base_model}. "
            "Run CPT first or point to base model."
        )

    return issues


def launch_axolotl(config_path: str | Path) -> int:
    """Launch Axolotl SFT training."""
    cmd = [
        sys.executable, "-m", "axolotl.cli.train",
        str(config_path),
    ]

    logger.info("Launching SFT: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=False,
        check=False,
    )

    return result.returncode


def run_sft(
    config_path: str | Path | None = None,
    dry_run: bool = False,
    validate_data: bool = True,
) -> int:
    """Main entry point for SFT.

    Args:
        config_path: Path to config.
        dry_run: Validate only.
        validate_data: Run trajectory validation (recommended).

    Returns:
        0 on success.
    """
    config = load_config(config_path)

    if validate_data:
        issues = validate_prerequisites(config)
        if issues:
            for issue in issues:
                logger.warning("SFT issue: %s", issue)
            if any("format consistent: False" in i for i in issues):
                logger.error("FATAL: Tool call format inconsistency detected. Fix data before training.")
                return 1

    if dry_run:
        logger.info("Dry run complete.")
        return 0

    actual_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return launch_axolotl(actual_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch SFT training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    sys.exit(run_sft(args.config, args.dry_run, not args.skip_validation))
