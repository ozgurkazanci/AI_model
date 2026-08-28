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
            assert ".subckt" in netlist
            assert ".ends" in netlist


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
