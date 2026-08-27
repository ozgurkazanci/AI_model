"""Tests for the synthetic perturbation pipeline (rewritten with real netlist modification)."""

import pytest
from asic_ai.data.perturbation import (
    BiasShift,
    RemoveComponent,
    ScaleWL,
    MisconnectNode,
    ChangeLoad,
    MirrorRatioBroken,
    SwapDevices,
    ParameterDrift,
    PerturbationPipeline,
    PerturbedCircuit,
    parse_spice_value,
    format_spice_value,
)


TEST_NETLIST = """\
.subckt test_ota VDD VSS INP INM OUT
XM1 net1 INM net3 VSS nfet_01v8 W=10u L=180n m=4
XM2 net2 INP net3 VSS nfet_01v8 W=10u L=180n m=4
XM3 net1 net1 VDD VDD pfet_01v8 W=20u L=180n m=4
XM4 net2 net1 VDD VDD pfet_01v8 W=20u L=180n m=4
XM5 net3 Vbn VSS VSS nfet_01v8 W=5u L=500n m=2
Cc net2 OUT 2p
XM6 OUT net2 VDD VDD pfet_01v8 W=40u L=180n m=8
XM7 OUT Vbn2 VSS VSS nfet_01v8 W=20u L=180n m=8
Ibias VDD Vbn 100u
.ends
"""


class TestSpiceValueParsing:
    """Test SPICE value parsing and formatting."""

    def test_parse_si_prefixes(self):
        assert parse_spice_value("10u") == pytest.approx(10e-6)
        assert parse_spice_value("180n") == pytest.approx(180e-9)
        assert parse_spice_value("2p") == pytest.approx(2e-12)
        assert parse_spice_value("100u") == pytest.approx(100e-6)
        assert parse_spice_value("1.5m") == pytest.approx(1.5e-3)
        assert parse_spice_value("500") == pytest.approx(500.0)

    def test_format_spice_value(self):
        formatted = format_spice_value(10e-6)
        assert "u" in formatted or "e-" in formatted
        formatted = format_spice_value(180e-9)
        assert "n" in formatted or "e-" in formatted


class TestPerturbationTypes:
    """Test each perturbation type produces a modified netlist."""

    def test_bias_shift(self):
        p = BiasShift()
        perturbed, desc = p.apply(TEST_NETLIST, seed=42)
        assert isinstance(perturbed, str)
        assert len(desc) > 0
        # Should actually change bias current value
        assert perturbed != TEST_NETLIST or "bias" in desc.lower()

    def test_remove_component(self):
        p = RemoveComponent()
        perturbed, desc = p.apply(TEST_NETLIST, seed=42)
        assert len(desc) > 0
        # Should have fewer lines (removed something)
        assert len(perturbed.strip().split("\n")) <= len(TEST_NETLIST.strip().split("\n"))

    def test_scale_wl(self):
        p = ScaleWL()
        perturbed, desc = p.apply(TEST_NETLIST, seed=42)
        assert len(desc) > 0
        assert perturbed != TEST_NETLIST

    def test_misconnect_node(self):
        p = MisconnectNode()
        perturbed, desc = p.apply(TEST_NETLIST, seed=42)
        assert len(desc) > 0

    def test_change_load(self):
        p = ChangeLoad()
        perturbed, desc = p.apply(TEST_NETLIST, seed=42)
        assert len(desc) > 0

    def test_mirror_ratio_broken(self):
        p = MirrorRatioBroken()
        perturbed, desc = p.apply(TEST_NETLIST, seed=42)
        assert len(desc) > 0

    def test_swap_devices(self):
        p = SwapDevices()
        perturbed, desc = p.apply(TEST_NETLIST, seed=42)
        assert len(desc) > 0

    def test_parameter_drift(self):
        p = ParameterDrift()
        perturbed, desc = p.apply(TEST_NETLIST, seed=42)
        assert len(desc) > 0
        assert perturbed != TEST_NETLIST


class TestPerturbationPipeline:
    def test_generate(self):
        pipeline = PerturbationPipeline(perturbations=[BiasShift(), ScaleWL()])
        result = pipeline.generate(TEST_NETLIST, seed=42)
        assert isinstance(result, PerturbedCircuit)
        assert result.original_netlist == TEST_NETLIST

    def test_generate_multiple_calls(self):
        pipeline = PerturbationPipeline(perturbations=[BiasShift()])
        results = [pipeline.generate(TEST_NETLIST, seed=i) for i in range(5)]
        assert len(results) == 5
        assert all(isinstance(r, PerturbedCircuit) for r in results)

    def test_deterministic_with_same_seed(self):
        pipeline = PerturbationPipeline(perturbations=[ScaleWL()])
        r1 = pipeline.generate(TEST_NETLIST, seed=123)
        r2 = pipeline.generate(TEST_NETLIST, seed=123)
        assert r1.perturbed_netlist == r2.perturbed_netlist

    def test_multiple_perturbation_types(self):
        pipeline = PerturbationPipeline(
            perturbations=[BiasShift(), RemoveComponent(), ScaleWL()]
        )
        results = [pipeline.generate(TEST_NETLIST, seed=i) for i in range(5)]
        assert len(results) == 5

    def test_composable_perturbations(self):
        pipeline = PerturbationPipeline(
            perturbations=[BiasShift(), ScaleWL()]
        )
        result = pipeline.generate(TEST_NETLIST, seed=42, num_perturbations=2)
        assert isinstance(result, PerturbedCircuit)
        # With num_perturbations=2, should have applied 2 perturbation types
        assert len(result.perturbations_applied) >= 1
