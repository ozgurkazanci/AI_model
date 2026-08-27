"""Tokenizer extension for circuit design domain.

Extends a base tokenizer with domain-specific tokens following the design document
(Section 3.1). Strategy: LIMITED extension, NOT full replacement.

Added tokens:
- SI prefixes as single tokens: f, p, n, u, m, k, M, G, T
- Common unit combinations: uA, pF, MHz, dB, V/V, deg, mV, nA, fF, GHz, Ohm, kOhm, MOhm
- Device names: nfet_01v8, pfet_01v8, sky130_fd_pr_* variants
- Netlist keywords: .subckt, .ends, .tran, .ac, .dc, .noise, .measure, .param, .include, .lib
- Common circuit terms: VDD, VSS, GND, Vbias, Ibias, gm, gds, Cgs, Cgd, Cdb

What is NOT done: Full digit-based tokenization change — would destroy base model knowledge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenExtensionConfig:
    """Configuration for tokenizer extension."""
    # SI prefixes
    si_prefixes: list[str] = field(default_factory=lambda: [
        "f", "p", "n", "u", "µ", "m", "k", "M", "G", "T",
    ])

    # Unit combinations (most frequent in circuit design)
    unit_combinations: list[str] = field(default_factory=lambda: [
        # Current
        "fA", "pA", "nA", "uA", "µA", "mA", "A",
        # Voltage
        "fV", "pV", "nV", "uV", "µV", "mV", "V", "kV",
        # Capacitance
        "fF", "pF", "nF", "uF", "µF", "mF", "F",
        # Resistance
        "mOhm", "Ohm", "kOhm", "MOhm", "GOhm",
        "mΩ", "Ω", "kΩ", "MΩ", "GΩ",
        # Frequency
        "Hz", "kHz", "MHz", "GHz", "THz",
        # Power
        "fW", "pW", "nW", "uW", "µW", "mW", "W", "kW",
        # Gain/ratio
        "dB", "V/V", "A/A", "dBm", "dBV",
        # Phase/angle
        "deg", "rad",
        # Time
        "fs", "ps", "ns", "us", "µs", "ms", "s",
        # Transconductance
        "uS", "mS", "S",
        # Temperature
        "degC", "degK",
    ])

    # sky130 device names
    sky130_devices: list[str] = field(default_factory=lambda: [
        "nfet_01v8", "pfet_01v8",
        "nfet_01v8_lvt", "pfet_01v8_lvt",
        "nfet_01v8_hvt", "pfet_01v8_hvt",
        "nfet_03v3_nvt", "nfet_05v0_nvt",
        "pfet_g5v0d10v5",
        "sky130_fd_pr__nfet_01v8",
        "sky130_fd_pr__pfet_01v8",
        "sky130_fd_pr__nfet_01v8_lvt",
        "sky130_fd_pr__pfet_01v8_lvt",
        "sky130_fd_pr__nfet_01v8_hvt",
        "sky130_fd_pr__pfet_01v8_hvt",
        "sky130_fd_pr__res_generic_nd",
        "sky130_fd_pr__res_generic_pd",
        "sky130_fd_pr__cap_mim_m3_1",
        "sky130_fd_pr__cap_mim_m3_2",
        "sky130_fd_pr__diode_pw2nd_05v5",
    ])

    # SPICE netlist keywords
    netlist_keywords: list[str] = field(default_factory=lambda: [
        ".subckt", ".ends", ".end",
        ".tran", ".ac", ".dc", ".noise", ".op", ".pz",
        ".measure", ".meas", ".param", ".func",
        ".include", ".lib", ".model", ".option", ".options",
        ".save", ".print", ".plot",
        ".ic", ".nodeset",
        ".global", ".temp",
        ".control", ".endc",
        ".mc", ".alter",
        # nabla-specific (future)
        ".stb", ".corners", ".pvt",
    ])

    # Common circuit design terms
    circuit_terms: list[str] = field(default_factory=lambda: [
        # Power/ground
        "VDD", "VSS", "GND", "AVDD", "DVDD", "AVSS", "DVSS",
        # Bias
        "Vbias", "Ibias", "Vbn", "Vbp", "Vcm", "Vref",
        # Small-signal parameters
        "gm", "gds", "gmb", "gm/ID", "gm/Id",
        "Cgs", "Cgd", "Cdb", "Csb", "Cgb",
        "ft", "fT", "fmax",
        "Vth", "Vov", "VGS", "VDS", "VBS",
        # Performance metrics
        "UGB", "GBW", "UGBW",
        "CMRR", "PSRR", "ICMR",
        "THD", "SNR", "SFDR", "ENOB",
        "PM", "GM",  # phase margin, gain margin
        "SR",  # slew rate
        # Topology terms
        "OTA", "LDO", "BGR", "PLL", "ADC", "DAC",
        "OPAMP", "TIA", "VCO", "LNA",
        # Process
        "NMOS", "PMOS", "CMOS",
        "nfet", "pfet", "nmos", "pmos",
        "W/L", "W", "L", "M",  # multiplier
    ])

    # GF180MCU device names (another open PDK)
    gf180_devices: list[str] = field(default_factory=lambda: [
        "nfet_03v3", "pfet_03v3",
        "nfet_06v0", "pfet_06v0",
        "nfet_03v3_dss", "pfet_03v3_dss",
    ])


def get_new_tokens(config: TokenExtensionConfig | None = None) -> list[str]:
    """Get the complete list of new tokens to add to the tokenizer.

    Args:
        config: Token extension configuration. Uses defaults if None.

    Returns:
        Deduplicated list of new tokens.
    """
    if config is None:
        config = TokenExtensionConfig()

    all_tokens: list[str] = []
    all_tokens.extend(config.si_prefixes)
    all_tokens.extend(config.unit_combinations)
    all_tokens.extend(config.sky130_devices)
    all_tokens.extend(config.netlist_keywords)
    all_tokens.extend(config.circuit_terms)
    all_tokens.extend(config.gf180_devices)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for token in all_tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)

    return unique_tokens


def extend_tokenizer(
    tokenizer_name_or_path: str,
    output_path: str | Path,
    config: TokenExtensionConfig | None = None,
    test_strings: list[str] | None = None,
) -> dict[str, Any]:
    """Extend a HuggingFace tokenizer with circuit design tokens.

    This function:
    1. Loads the base tokenizer
    2. Adds new domain-specific tokens
    3. Saves the extended tokenizer
    4. Optionally tests tokenization on sample strings

    Args:
        tokenizer_name_or_path: HuggingFace model ID or local path.
        output_path: Where to save the extended tokenizer.
        config: Token extension configuration.
        test_strings: Optional strings to test tokenization before/after.

    Returns:
        Dictionary with statistics about the extension.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers package required for tokenizer extension. "
            "Install with: pip install transformers"
        ) from e

    output_path = Path(output_path)
    new_tokens = get_new_tokens(config)

    # Load base tokenizer
    logger.info("Loading base tokenizer: %s", tokenizer_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    original_vocab_size = len(tokenizer)

    # Test tokenization before extension
    before_results: dict[str, list[str]] = {}
    if test_strings:
        for s in test_strings:
            tokens = tokenizer.tokenize(s)
            before_results[s] = tokens

    # Filter out tokens that already exist
    tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    logger.info(
        "Adding %d new tokens (out of %d candidates, %d already exist)",
        len(tokens_to_add),
        len(new_tokens),
        len(new_tokens) - len(tokens_to_add),
    )

    # Add new tokens
    num_added = tokenizer.add_tokens(tokens_to_add)
    new_vocab_size = len(tokenizer)

    # Test tokenization after extension
    after_results: dict[str, list[str]] = {}
    if test_strings:
        for s in test_strings:
            tokens = tokenizer.tokenize(s)
            after_results[s] = tokens

    # Save extended tokenizer
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))

    # Build statistics
    stats = {
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "tokens_added": num_added,
        "tokens_requested": len(new_tokens),
        "tokens_already_existed": len(new_tokens) - len(tokens_to_add),
        "output_path": str(output_path),
    }

    if test_strings:
        stats["tokenization_comparison"] = {
            s: {
                "before": before_results.get(s, []),
                "after": after_results.get(s, []),
                "before_count": len(before_results.get(s, [])),
                "after_count": len(after_results.get(s, [])),
                "improvement": (
                    len(before_results.get(s, [])) - len(after_results.get(s, []))
                ),
            }
            for s in test_strings
        }

    # Save stats
    stats_path = output_path / "extension_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Tokenizer extended: %d → %d tokens", original_vocab_size, new_vocab_size)
    return stats


# Default test strings for validating tokenization improvement
DEFAULT_TEST_STRINGS = [
    # Numbers with SI prefixes
    "W=4.2u L=180n",
    "Ibias=200uA",
    "Cc=2pF",
    "gm=1.5mS",
    "ft=5GHz",
    # Specs
    "dc_gain=60dB UGB=50MHz PM=60deg",
    "PSRR=40dB CMRR=80dB",
    "Idd=200uA Vout=1.2V",
    # Device instantiation
    "XM1 nfet_01v8 W=10u L=180n M=4",
    "sky130_fd_pr__nfet_01v8_lvt",
    # Netlist
    ".subckt ota_2stage VDD VSS INP INM OUT",
    ".tran 1n 100u",
    ".ac dec 100 1 10G",
    ".measure ac gain_db MAX vdb(out)",
    # gm/ID methodology
    "gm/ID=15 V/V, Vov=100mV",
]
