"""Tests for the tokenizer extension module."""

import pytest
from asic_ai.tokenizer.extend import (
    TokenExtensionConfig,
    get_new_tokens,
    DEFAULT_TEST_STRINGS,
)


class TestTokenExtensionConfig:
    def test_default_config(self):
        config = TokenExtensionConfig()
        assert len(config.si_prefixes) > 0
        assert len(config.unit_combinations) > 0
        assert len(config.sky130_devices) > 0
        assert len(config.netlist_keywords) > 0
        assert len(config.circuit_terms) > 0

    def test_custom_si_prefixes(self):
        config = TokenExtensionConfig(si_prefixes=["n", "u", "m"])
        assert config.si_prefixes == ["n", "u", "m"]


class TestGetNewTokens:
    def test_returns_list(self):
        tokens = get_new_tokens()
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_deduplication(self):
        tokens = get_new_tokens()
        assert len(tokens) == len(set(tokens)), "Tokens should be deduplicated"

    def test_si_prefixes_included(self):
        tokens = get_new_tokens()
        for prefix in ["f", "p", "n", "u", "m", "k", "M", "G", "T"]:
            assert prefix in tokens, f"SI prefix '{prefix}' should be included"

    def test_device_names_included(self):
        tokens = get_new_tokens()
        assert "nfet_01v8" in tokens
        assert "pfet_01v8" in tokens

    def test_netlist_keywords_included(self):
        tokens = get_new_tokens()
        for kw in [".subckt", ".tran", ".ac", ".measure"]:
            assert kw in tokens, f"Netlist keyword '{kw}' should be included"

    def test_circuit_terms_included(self):
        tokens = get_new_tokens()
        for term in ["VDD", "VSS", "gm", "gds", "Cgs"]:
            assert term in tokens, f"Circuit term '{term}' should be included"

    def test_unit_combinations_included(self):
        tokens = get_new_tokens()
        for unit in ["uA", "pF", "MHz", "dB", "V/V", "deg"]:
            assert unit in tokens, f"Unit combination '{unit}' should be included"

    def test_custom_config(self):
        config = TokenExtensionConfig(
            si_prefixes=["n", "u"],
            unit_combinations=["pF"],
            sky130_devices=[],
            netlist_keywords=[],
            circuit_terms=[],
            gf180_devices=[],
        )
        tokens = get_new_tokens(config)
        assert "n" in tokens
        assert "u" in tokens
        assert "pF" in tokens
        assert "nfet_01v8" not in tokens  # excluded


class TestDefaultTestStrings:
    def test_default_strings_exist(self):
        assert isinstance(DEFAULT_TEST_STRINGS, list)
        assert len(DEFAULT_TEST_STRINGS) > 0

    def test_contains_si_examples(self):
        all_strings = " ".join(DEFAULT_TEST_STRINGS)
        assert "180n" in all_strings or "4.2u" in all_strings
