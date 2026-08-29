"""Tests for new CLI scripts: chat, benchmark, training_monitor, merge_lora, run_eval."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestTrainingMonitor:
    """Test training_monitor.py functions."""

    def test_format_loss_chart_empty(self):
        """Chart with no data returns message."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from training_monitor import format_loss_chart
        result = format_loss_chart([])
        assert "No loss data" in result

    def test_format_loss_chart_with_data(self):
        """Chart with data shows loss values."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from training_monitor import format_loss_chart
        log_history = [
            {"step": 10, "loss": 2.0},
            {"step": 20, "loss": 1.0},
            {"step": 30, "loss": 0.5},
        ]
        result = format_loss_chart(log_history)
        assert "2.0000" in result
        assert "0.5000" in result
        assert "Improvement" in result

    def test_read_trainer_state_missing(self):
        """Returns None when no trainer state found."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from training_monitor import read_trainer_state
        result = read_trainer_state("/nonexistent/path")
        assert result is None


class TestPrepareTrainingData:
    """Test prepare_training_data.py functions."""

    def test_estimate_difficulty(self):
        """Difficulty estimation is bounded 0-1."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from prepare_training_data import estimate_difficulty

        easy = {"messages": [{"role": "user", "content": "hi"}], "score": 0.95}
        hard = {"messages": [{"role": "user", "content": "x"}] * 20, "score": 0.3}

        d_easy = estimate_difficulty(easy)
        d_hard = estimate_difficulty(hard)

        assert 0 <= d_easy <= 1
        assert 0 <= d_hard <= 1
        assert d_easy < d_hard  # Easy should have lower difficulty

    def test_generate_pdk_corners_examples(self):
        """Generate corners examples for the last missing tool."""
        import random
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from prepare_training_data import generate_pdk_corners_examples

        examples = generate_pdk_corners_examples(random.Random(42), count=3)
        assert len(examples) == 3
        for ex in examples:
            assert "messages" in ex
            assert ex["primary_tool"] == "pdk.get_corners"


class TestTemplatesCoverage:
    """Extended template tests for new templates."""

    def test_source_follower_exists(self):
        from asic_ai.data.templates import get_template
        t = get_template("source_follower")
        assert t.category == "analog"
        assert "W_M1" in t.parameters

    def test_diff_pair_exists(self):
        from asic_ai.data.templates import get_template
        t = get_template("diff_pair")
        assert t.category == "analog"
        assert "W_IN" in t.parameters

    def test_ring_osc_exists(self):
        from asic_ai.data.templates import get_template
        t = get_template("ring_osc")
        assert t.category == "digital"
        assert "WP" in t.parameters

    def test_total_template_count(self):
        from asic_ai.data.templates import list_templates
        assert len(list_templates()) == 17

    def test_all_have_typical_specs(self):
        from asic_ai.data.templates import list_templates
        for t in list_templates():
            assert t.typical_specs, f"{t.id} missing typical_specs"

    def test_digital_templates(self):
        from asic_ai.data.templates import list_templates
        digital = list_templates(category="digital")
        assert len(digital) >= 1  # ring_osc at minimum


class TestSFTDataIntegrity:
    """Validate the prepared training data."""

    def test_train_final_exists(self):
        assert Path("data/sft/train_final.jsonl").exists()

    def test_val_final_exists(self):
        assert Path("data/sft/val_final.jsonl").exists()

    def test_train_final_valid_json(self):
        with open("data/sft/train_final.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                assert "messages" in data, f"Line {i} missing messages"

    def test_train_has_enough_examples(self):
        count = 0
        with open("data/sft/train_final.jsonl", encoding="utf-8") as f:
            for line in f:
                count += 1
        assert count >= 300, f"Only {count} examples, expected >= 300"

    def test_format_validation(self):
        """All training examples pass format validation."""
        from asic_ai.data.format import validate_sft_format
        errors_found = 0
        with open("data/sft/train_final.jsonl", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                is_valid, errors = validate_sft_format(data["messages"])
                if not is_valid:
                    errors_found += 1
        assert errors_found == 0, f"{errors_found} invalid examples"
