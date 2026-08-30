"""Tests for circuit topology template library."""
import pytest

from asic_ai.data.templates import (
    CircuitTemplate,
    TEMPLATES,
    get_template,
    list_templates,
    render_template,
)


class TestTemplateRegistry:
    def test_templates_exist(self):
        assert len(TEMPLATES) >= 6

    def test_all_have_required_fields(self):
        for tid, t in TEMPLATES.items():
            assert t.id == tid
            assert t.name
            assert t.category in ("analog", "digital")
            assert t.netlist
            assert t.parameters

    def test_get_template(self):
        t = get_template("ota_2stage")
        assert t.name == "Two-Stage Miller OTA"

    def test_get_template_not_found(self):
        with pytest.raises(KeyError):
            get_template("nonexistent")

    def test_list_templates_all(self):
        all_t = list_templates()
        assert len(all_t) >= 6

    def test_list_templates_analog(self):
        analog = list_templates(category="analog")
        assert all(t.category == "analog" for t in analog)
        assert len(analog) >= 6


class TestTemplateRendering:
    def test_render_ota_default(self):
        netlist = render_template("ota_2stage")
        assert ".subckt ota_2stage" in netlist
        assert "nfet_01v8" in netlist
        assert "pfet_01v8" in netlist

    def test_render_ota_custom(self):
        netlist = render_template("ota_2stage", w1=20, cc=5)
        assert "W=20u" in netlist
        assert "5p" in netlist

    def test_render_bandgap(self):
        netlist = render_template("bandgap_brokaw")
        assert ".subckt bandgap" in netlist
        assert "pnp_01v8" in netlist

    def test_render_ldo(self):
        netlist = render_template("ldo_basic")
        assert ".subckt ldo" in netlist
        assert "pfet_01v8" in netlist

    def test_render_current_mirror(self):
        netlist = render_template("current_mirror_cascode")
        assert ".subckt cm_cascode" in netlist

    def test_render_comparator(self):
        netlist = render_template("comparator_basic")
        assert ".subckt comparator" in netlist

    def test_all_templates_renderable(self):
        for tid in TEMPLATES:
            netlist = render_template(tid)
            # Subcircuit templates use .subckt/.ends, flat templates use .end
            assert ".subckt" in netlist or ".end" in netlist


class TestTemplateSpecs:
    def test_ota_has_specs(self):
        t = get_template("ota_2stage")
        assert "dc_gain" in t.typical_specs
        assert "ugb" in t.typical_specs

    def test_bandgap_has_specs(self):
        t = get_template("bandgap_brokaw")
        assert "vref" in t.typical_specs
        assert "tc" in t.typical_specs

    def test_ldo_has_specs(self):
        t = get_template("ldo_basic")
        assert "dropout" in t.typical_specs
        assert "psrr" in t.typical_specs


class TestNewTemplates:
    """Tests for Phase 58 templates: PLL, ADC, DAC, LNA."""

    def test_pll_exists(self):
        t = get_template("charge_pump_pll")
        assert t.category == "analog"
        assert "lock_time" in t.typical_specs
        assert "jitter" in t.typical_specs

    def test_flash_adc_exists(self):
        t = get_template("flash_adc_3bit")
        assert t.category == "analog"
        assert "resolution" in t.typical_specs
        assert "dnl" in t.typical_specs

    def test_r2r_dac_exists(self):
        t = get_template("r2r_dac_4bit")
        assert t.category == "analog"
        assert "resolution" in t.typical_specs
        assert "settling_time" in t.typical_specs

    def test_cg_lna_exists(self):
        t = get_template("cg_lna")
        assert t.category == "analog"
        assert "gain" in t.typical_specs
        assert "nf" in t.typical_specs
        assert "s11" in t.typical_specs

    def test_pll_render(self):
        rendered = render_template("charge_pump_pll", wp=20, wn=10)
        assert "20" in rendered
        assert "charge_pump_pll" in rendered

    def test_all_new_templates_have_design_notes(self):
        for tid in ["charge_pump_pll", "flash_adc_3bit", "r2r_dac_4bit", "cg_lna"]:
            t = get_template(tid)
            assert t.design_notes, f"{tid} missing design_notes"


class TestPDKKnowledge:
    """Tests for PDK knowledge base."""

    def test_sky130_params(self):
        from asic_ai.data.pdk_knowledge import get_pdk_params
        p = get_pdk_params("sky130")
        assert p["process_name"] == "sky130"
        assert p["supply_voltage"] == 1.8
        assert p["nmos"]["vth0"] == pytest.approx(0.49, abs=0.1)
        assert p["pmos"]["vth0"] == pytest.approx(-0.54, abs=0.1)

    def test_gf180mcu_params(self):
        from asic_ai.data.pdk_knowledge import get_pdk_params
        p = get_pdk_params("gf180mcu")
        assert p["supply_voltage"] == 3.3

    def test_unknown_pdk_raises(self):
        from asic_ai.data.pdk_knowledge import get_pdk_params
        with pytest.raises(KeyError):
            get_pdk_params("unknown_pdk")

    def test_calculate_gm(self):
        from asic_ai.data.pdk_knowledge import calculate_gm
        gm = calculate_gm(10e-6, 0.5e-6, 100e-6)
        assert gm > 0
        assert gm < 10e-3  # reasonable range

    def test_calculate_mismatch(self):
        from asic_ai.data.pdk_knowledge import calculate_mismatch_sigma
        sig = calculate_mismatch_sigma(10e-6, 0.5e-6)
        assert sig > 0
        assert sig < 50e-3  # reasonable mV range

    def test_sky130_corners(self):
        from asic_ai.data.pdk_knowledge import get_pdk_params
        p = get_pdk_params("sky130")
        corners = p["corners"]
        assert "tt" in corners
        assert "ss" in corners
        assert "ff" in corners
        assert corners["tt"]["temperature"] == 27
        # SIGN-OFF CONVENTION: SS is slow process, LOW supply and HOT; FF is
        # fast process, HIGH supply and COLD. This assertion used to demand
        # ss == -40, which encoded the inverted convention that made every
        # corner partially cancel itself.
        assert corners["ss"]["temperature"] == 125
        assert corners["ss"]["voltage"] < corners["tt"]["voltage"]
        assert corners["ff"]["temperature"] == -40
        assert corners["ff"]["voltage"] > corners["tt"]["voltage"]

