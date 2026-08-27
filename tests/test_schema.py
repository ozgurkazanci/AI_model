"""Tests for the tool interface schema module."""

import json
import pytest
from asic_ai.tool_interface.schema import (
    ActionType,
    SimParams,
    PVTCorner,
    DCResult,
    ACResult,
    TranResult,
    StabilityResult,
    CornerResult,
    MonteCarloResult,
    NetlistPatch,
    NetlistPatchOperation,
    LintResult,
    LintError,
    SpecCheckResult,
    SpecCheckDetail,
    DeviceQueryResult,
    AgentAction,
    AgentObservation,
    SignalData,
    get_json_schema,
)


class TestResultDataclasses:
    """Test that all result models can be instantiated with valid data."""

    def test_dc_result(self):
        r = DCResult(op_points={"v(out)": 0.9, "i(vdd)": -200e-6})
        assert r.op_points["v(out)"] == 0.9

    def test_ac_result(self):
        r = ACResult(
            frequencies=[1.0, 10.0, 100.0],
            signals={
                "v(out)": SignalData(name="v(out)", x_values=[1.0, 10.0, 100.0], y_values=[60.0, 55.0, 40.0])
            },
        )
        assert len(r.frequencies) == 3

    def test_tran_result(self):
        r = TranResult(
            time=[0.0, 1e-9, 2e-9],
            signals={"v(out)": SignalData(name="v(out)", x_values=[0.0, 1e-9, 2e-9], y_values=[0.0, 0.5, 0.9])},
        )
        assert len(r.time) == 3

    def test_stability_result(self):
        r = StabilityResult(
            phase_margin=65.0,
            gain_margin=12.0,
            loop_gain=SignalData(name="loop_gain", x_values=[1.0], y_values=[60.0]),
        )
        assert r.phase_margin == 65.0

    def test_corner_result(self):
        r = CornerResult(
            corner=PVTCorner(process="tt", voltage=1.8, temperature=27.0),
            dc=DCResult(op_points={"v(out)": 0.9}),
        )
        assert r.corner.process == "tt"

    def test_monte_carlo_result(self):
        r = MonteCarloResult(seed=42, runs=100, results=[{"gain_db": 60.0}])
        assert r.runs == 100

    def test_netlist_patch(self):
        patch = NetlistPatch(
            operations=[
                NetlistPatchOperation(op="modify_param", target="XM1", value="W=20u"),
            ]
        )
        assert len(patch.operations) == 1

    def test_lint_result(self):
        r = LintResult(
            errors=[LintError(node="net5", message="Floating node", severity="error")],
            passed=False,
        )
        assert r.passed is False

    def test_spec_check_result(self):
        r = SpecCheckResult(
            score=0.85,
            breakdown={
                "gain_db": SpecCheckDetail(min_value=60.0, actual=65.0, met=True, score=0.9),
            },
        )
        assert r.score == 0.85

    def test_device_query_result(self):
        r = DeviceQueryResult(
            model="nfet_01v8", W=10e-6, L=180e-9,
            VGS=0.6, VDS=0.9, VSB=0.0,
            gm=1.5e-3, gds=50e-6, id=200e-6,
            ft=5e9, cgs=10e-15, cgd=2e-15, cdb=5e-15,
            vth=0.45, region="saturation",
        )
        assert r.region == "saturation"


class TestPVTCorner:
    def test_valid_corner(self):
        c = PVTCorner(process="ss", voltage=1.62, temperature=125.0)
        assert c.process == "ss"

    def test_multiple_corners(self):
        corners = [
            PVTCorner(process="tt", voltage=1.8, temperature=27.0),
            PVTCorner(process="ss", voltage=1.62, temperature=125.0),
            PVTCorner(process="ff", voltage=1.98, temperature=-40.0),
        ]
        assert len(corners) == 3


class TestSimParams:
    def test_sim_params(self):
        p = SimParams(
            analysis_type="ac",
            start=1.0,
            stop=10e9,
            points=100,
        )
        assert p.analysis_type == "ac"

    def test_sim_params_with_options(self):
        p = SimParams(
            analysis_type="tran",
            start=0.0,
            stop=100e-6,
            step=1e-9,
            options={"method": "gear"},
        )
        assert p.options["method"] == "gear"


class TestAgentAction:
    def test_valid_action(self):
        a = AgentAction(
            action_type=ActionType.SIMULATE,
            arguments={"netlist": "...", "params": {}},
        )
        assert a.action_type == ActionType.SIMULATE

    def test_action_enum_values(self):
        assert ActionType.SIMULATE.value == "SIMULATE"
        assert ActionType.MEASURE.value == "MEASURE"
        assert ActionType.QUERY_PDK.value == "QUERY_PDK"
        assert ActionType.PATCH_NETLIST.value == "PATCH_NETLIST"
        assert ActionType.LINT.value == "LINT"
        assert ActionType.SET_SPEC.value == "SET_SPEC"
        assert ActionType.OPTIMIZE.value == "OPTIMIZE"


class TestAgentObservation:
    def test_valid_observation(self):
        obs = AgentObservation(
            netlist_state=".subckt test ...",
            last_results={"gain_db": 60.5},
            spec_status=SpecCheckResult(
                score=0.8,
                breakdown={"gain_db": SpecCheckDetail(min_value=60.0, actual=60.5, met=True, score=0.8)},
            ),
            step_count=5,
        )
        assert obs.step_count == 5
        assert obs.spec_status.score == 0.8


class TestJsonSchemaExport:
    def test_json_schema_export(self):
        schema_str = get_json_schema()
        schemas = json.loads(schema_str)
        assert "SimParams" in schemas
        assert "AgentAction" in schemas
        assert "DCResult" in schemas
        # Check that AgentAction schema has action_type
        assert "action_type" in schemas["AgentAction"]["properties"]


class TestValidation:
    def test_invalid_action_type_rejected(self):
        with pytest.raises(Exception):
            AgentAction(action_type="invalid_type", arguments={})

    def test_missing_required_field_rejected(self):
        with pytest.raises(Exception):
            SimParams()  # analysis_type is required
