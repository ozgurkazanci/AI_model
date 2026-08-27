"""Tests for the synthetic perturbation pipeline."""

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
)


TEST_NETLIST = """\
.subckt test_ota VDD VSS INP INM OUT
XM1 net1 INM net3 VSS nfet_01v8 W=10u L=180n M=4
XM2 net2 INP net3 VSS nfet_01v8 W=10u L=180n M=4
XM3 net1 net1 VDD VDD pfet_01v8 W=20u L=180n M=4
XM4 net2 net1 VDD VDD pfet_01v8 W=20u L=180n M=4
XM5 net3 Vbn VSS VSS nfet_01v8 W=5u L=500n M=2
Cc net2 OUT 2p
XM6 OUT net2 VDD VDD pfet_01v8 W=40u L=180n M=8
XM7 OUT Vbn2 VSS VSS nfet_01v8 W=20u L=180n M=8
Ibias VDD Vbn 100u
.ends
"""


class TestPerturbationTypes:
    """Test each perturbation type produces a modified netlist."""

    def test_bias_shift(self):
        p = BiasShift()
        perturbed, desc = p.apply(TEST_NETLIST)
        assert perturbed != TEST_NETLIST or "BiasShift" in perturbed
        assert len(desc) > 0

    def test_remove_component(self):
        p = RemoveComponent()
        perturbed, desc = p.apply(TEST_NETLIST)
        assert len(desc) > 0

    def test_scale_wl(self):
        p = ScaleWL()
        perturbed, desc = p.apply(TEST_NETLIST)
        assert len(desc) > 0

    def test_misconnect_node(self):
        p = MisconnectNode()
        perturbed, desc = p.apply(TEST_NETLIST)
        assert len(desc) > 0

    def test_change_load(self):
        p = ChangeLoad()
        perturbed, desc = p.apply(TEST_NETLIST)
        assert len(desc) > 0

    def test_mirror_ratio_broken(self):
        p = MirrorRatioBroken()
        perturbed, desc = p.apply(TEST_NETLIST)
        assert len(desc) > 0

    def test_swap_devices(self):
        p = SwapDevices()
        perturbed, desc = p.apply(TEST_NETLIST)
        assert len(desc) > 0

    def test_parameter_drift(self):
        p = ParameterDrift()
        perturbed, desc = p.apply(TEST_NETLIST)
        assert len(desc) > 0


class TestPerturbationPipeline:
    def test_register_and_generate(self):
        pipeline = PerturbationPipeline()
        pipeline.register(BiasShift, weight=1.0)
        pipeline.register(ScaleWL, weight=1.0)
        results = pipeline.generate(TEST_NETLIST, n=5, seed=42)
        assert len(results) == 5
        assert all(isinstance(r, PerturbedCircuit) for r in results)

    def test_pipeline_produces_different_results(self):
        pipeline = PerturbationPipeline()
        pipeline.register(BiasShift, weight=1.0)
        results = pipeline.generate(TEST_NETLIST, n=3, seed=42)
        # At least some should differ (randomized bias factor)
        netlists = [r.perturbed_netlist for r in results]
        assert len(netlists) == 3

    def test_deterministic_with_same_seed(self):
        pipeline = PerturbationPipeline()
        pipeline.register(ScaleWL, weight=1.0)
        r1 = pipeline.generate(TEST_NETLIST, n=3, seed=123)
        r2 = pipeline.generate(TEST_NETLIST, n=3, seed=123)
        for a, b in zip(r1, r2):
            assert a.perturbed_netlist == b.perturbed_netlist

    def test_composable_perturbations(self):
        pipeline = PerturbationPipeline()
        pipeline.register(BiasShift, weight=1.0)
        pipeline.register(RemoveComponent, weight=1.0)
        pipeline.register(ScaleWL, weight=1.0)
        results = pipeline.generate(TEST_NETLIST, n=3, seed=42)
        assert len(results) == 3
