"""Tests for Cadence Spectre WSL adapter.

Tests path conversion, environment setup, and adapter creation.
Actual simulation tests require Cadence license (skipped if unavailable).
"""
from __future__ import annotations

import pytest
from asic_ai.adapters.spectre_wsl import (
    SpectreWSLAdapter,
    check_spectre_available,
    win_to_wsl_path,
    wsl_to_win_path,
    SPECTRE_BIN,
    WSL_DISTRO,
)
from asic_ai.adapters.base import AdapterConfig
from asic_ai.adapters import get_adapter


class TestPathConversion:
    """Test Windows <-> WSL path conversion."""

    def test_drive_letter_to_wsl(self):
        assert win_to_wsl_path("C:\\Users\\test\\file.scs") == "/mnt/c/Users/test/file.scs"

    def test_drive_letter_lowercase(self):
        result = win_to_wsl_path("D:\\data\\sim.scs")
        assert result == "/mnt/d/data/sim.scs"

    def test_unc_path_to_wsl(self):
        result = win_to_wsl_path("\\\\wsl.localhost\\Alma_EDA\\opt\\eda\\test.scs")
        assert result == "/opt/eda/test.scs"

    def test_forward_slash_passthrough(self):
        result = win_to_wsl_path("/opt/eda/spectre")
        assert result == "/opt/eda/spectre"

    def test_wsl_to_win(self):
        result = wsl_to_win_path("/opt/eda/cadence/SPECTRE241")
        assert result == f"\\\\wsl.localhost\\{WSL_DISTRO}/opt/eda/cadence/SPECTRE241"


class TestSpectreAdapter:
    """Test Spectre adapter creation and configuration."""

    def test_adapter_creation(self, tmp_path):
        config = AdapterConfig(binary_path="", work_dir=str(tmp_path))
        adapter = SpectreWSLAdapter(config)
        assert adapter is not None
        assert adapter._sim_count == 0

    def test_factory_spectre(self, tmp_path):
        adapter = get_adapter("spectre", binary_path="", work_dir=str(tmp_path))
        assert isinstance(adapter, SpectreWSLAdapter)

    def test_factory_spectre_wsl(self, tmp_path):
        adapter = get_adapter("spectre_wsl", binary_path="", work_dir=str(tmp_path))
        assert isinstance(adapter, SpectreWSLAdapter)

    def test_work_dir_created(self, tmp_path):
        work = tmp_path / "sim_output"
        config = AdapterConfig(binary_path="", work_dir=str(work))
        SpectreWSLAdapter(config)
        assert work.exists()


class TestSpectreAvailability:
    """Test Spectre availability detection."""

    def test_check_returns_bool(self):
        result = check_spectre_available()
        assert isinstance(result, bool)

    def test_spectre_bin_path(self):
        assert "spectre" in SPECTRE_BIN.lower()
        assert "64bit" in SPECTRE_BIN

    def test_wsl_distro(self):
        assert WSL_DISTRO == "Alma_EDA"
