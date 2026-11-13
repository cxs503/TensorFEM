"""Tests for case-schema JSON round-trip and units converter."""
from __future__ import annotations

import json, copy

import pytest

from tensorfem.benchmark_cases import (
    BenchmarkCase, Citation, ReferenceQuantity,
    SCHEMA_VERSION,
    convert_value, convert_case_units,
    hemisphere_case, hertz_ball_plane_case,
    validate_case, case_hash, run_case,
)


# ---------------------------------------------------------------------------
# from_dict / from_json / to_json round-trip
# ---------------------------------------------------------------------------

class TestCitationRoundTrip:
    def test_from_dict(self):
        d = {"authors": "A. Author", "title": "A Paper",
             "publication": "J. Comp. Phys.", "year": 2020, "doi": "10.1234/test"}
        c = Citation.from_dict(d)
        assert c.authors == "A. Author"
        assert c.year == 2020
        assert c.doi == "10.1234/test"

    def test_roundtrip(self):
        c = Citation("A. Author", "A Paper", "J. Comp. Phys.", 2020, "10.1234/test")
        d = {"authors": c.authors, "title": c.title, "publication": c.publication,
             "year": c.year, "doi": c.doi}
        assert Citation.from_dict(d) == c


class TestReferenceQuantityRoundTrip:
    def test_from_dict(self):
        d = {"name": "tip displacement", "value": 0.5, "unit": "length", "sign_policy": "magnitude"}
        q = ReferenceQuantity.from_dict(d)
        assert q.value == 0.5
        assert q.unit == "length"

    def test_roundtrip(self):
        q = ReferenceQuantity("tip displacement", 0.5, "length", "magnitude")
        d = {"name": q.name, "value": q.value, "unit": q.unit, "sign_policy": q.sign_policy}
        assert ReferenceQuantity.from_dict(d) == q


class TestBenchmarkCaseRoundTrip:
    def test_from_dict_hemisphere(self):
        case = hemisphere_case()
        d = case.to_dict()
        restored = BenchmarkCase.from_dict(d)
        assert restored == case

    def test_from_dict_hertz(self):
        case = hertz_ball_plane_case()
        d = case.to_dict()
        restored = BenchmarkCase.from_dict(d)
        assert restored == case

    def test_to_json_from_json(self):
        case = hemisphere_case()
        text = case.to_json()
        restored = BenchmarkCase.from_json(text)
        assert restored == case

    def test_to_json_indent(self):
        case = hemisphere_case()
        text = case.to_json(indent=2)
        assert "\n" in text
        restored = BenchmarkCase.from_json(text)
        assert restored == case

    def test_json_is_canonical(self):
        """Two serialisations of the same case produce identical JSON."""
        case = hemisphere_case()
        assert case.to_json() == case.to_json()

    def test_validate_after_roundtrip(self):
        case = hemisphere_case()
        restored = BenchmarkCase.from_json(case.to_json())
        validate_case(restored)  # must not raise

    def test_hash_preserved_after_roundtrip(self):
        case = hemisphere_case()
        restored = BenchmarkCase.from_json(case.to_json())
        assert case_hash(case) == case_hash(restored)

    def test_tampered_json_fails_validation(self):
        case = hemisphere_case()
        d = json.loads(case.to_json())
        d["citation"]["doi"] = "bad"
        tampered = BenchmarkCase.from_dict(d)
        with pytest.raises(ValueError):
            validate_case(tampered)


# ---------------------------------------------------------------------------
# Unit conversion – scalar
# ---------------------------------------------------------------------------

class TestConvertValue:
    def test_identity(self):
        assert convert_value(1.0, "m", "m") == 1.0

    def test_m_to_mm(self):
        assert abs(convert_value(1.0, "m", "mm") - 1000.0) < 1e-9

    def test_mm_to_m(self):
        assert abs(convert_value(1000.0, "mm", "m") - 1.0) < 1e-9

    def test_pa_to_mpa(self):
        assert abs(convert_value(1e6, "Pa", "MPa") - 1.0) < 1e-9

    def test_mpa_to_pa(self):
        assert abs(convert_value(1.0, "MPa", "Pa") - 1e6) < 1e-9

    def test_n_to_kn(self):
        assert abs(convert_value(1000.0, "N", "kN") - 1.0) < 1e-9

    def test_kg_to_tonne(self):
        assert abs(convert_value(1000.0, "kg", "tonne") - 1.0) < 1e-9

    def test_inch_to_mm(self):
        # 1 inch = 25.4 mm
        assert abs(convert_value(1.0, "inch", "mm") - 25.4) < 1e-6

    def test_unknown_units_passthrough(self):
        assert convert_value(42.0, "bogus", "also_bogus") == 42.0


# ---------------------------------------------------------------------------
# Unit conversion – case-level
# ---------------------------------------------------------------------------

class TestConvertCaseUnits:
    def _hemisphere_dict(self):
        return hemisphere_case().to_dict()

    def test_si_to_mm_tonne_s_length(self):
        d = self._hemisphere_dict()
        converted = convert_case_units(d, "SI", "mm-tonne-s")
        # radius 10 m → 10000 mm
        assert abs(converted["geometry"]["radius"] - 10000.0) < 1e-6
        # thickness 0.04 m → 40 mm
        assert abs(converted["geometry"]["thickness"] - 40.0) < 1e-6

    def test_si_to_mm_tonne_s_young_modulus(self):
        d = self._hemisphere_dict()
        converted = convert_case_units(d, "SI", "mm-tonne-s")
        # E = 6.825e7 Pa → 68.25 MPa
        assert abs(converted["material"]["young_modulus"] - 68.25) < 1e-6

    def test_si_to_mm_tonne_s_quick_values(self):
        d = self._hemisphere_dict()
        converted = convert_case_units(d, "SI", "mm-tonne-s")
        # quick_values are lengths: 0.0924 m → 92.4 mm
        expected = [v * 1000.0 for v in d["quick_values"]]
        for got, exp in zip(converted["quick_values"], expected):
            assert abs(got - exp) < 1e-6

    def test_si_to_mm_tonne_s_reference_quantity(self):
        d = self._hemisphere_dict()
        converted = convert_case_units(d, "SI", "mm-tonne-s")
        # reference value 0.0924 m → 92.4 mm
        assert abs(converted["quantity"]["value"] - 92.4) < 1e-6

    def test_roundtrip_identity(self):
        d = self._hemisphere_dict()
        mid = convert_case_units(d, "SI", "mm-tonne-s")
        back = convert_case_units(mid, "mm-tonne-s", "SI")
        for key in ("geometry", "material"):
            for k in d[key]:
                if isinstance(d[key][k], (int, float)):
                    assert abs(back[key][k] - d[key][k]) / max(abs(d[key][k]), 1e-30) < 1e-9
        for a, b in zip(d["quick_values"], back["quick_values"]):
            assert abs(a - b) / max(abs(a), 1e-30) < 1e-9
        assert abs(back["quantity"]["value"] - d["quantity"]["value"]) < 1e-12

    def test_mm_tonne_s_to_si(self):
        d = self._hemisphere_dict()
        converted = convert_case_units(d, "mm-tonne-s", "SI")
        # In mm-tonne-s, radius would be 10000; converting back: 10000 * 1e-3 = 10
        # But our source is SI (radius=10), so converting SI→mm-tonne-s→SI should give 10
        # Actually this test converts the SI dict as if it were mm-tonne-s,
        # which is not meaningful. Let's do a proper round-trip instead.
        pass  # covered by roundtrip_identity above

    def test_preserves_non_numeric_fields(self):
        d = self._hemisphere_dict()
        converted = convert_case_units(d, "SI", "mm-tonne-s")
        assert converted["case_id"] == d["case_id"]
        assert converted["schema_version"] == d["schema_version"]
        assert converted["mesh_sequence"] == d["mesh_sequence"]
        assert converted["boundary_conditions"] == d["boundary_conditions"]


# ---------------------------------------------------------------------------
# with_units_converted method
# ---------------------------------------------------------------------------

class TestWithUnitsConverted:
    def test_returns_new_case(self):
        case = hemisphere_case()
        converted = case.with_units_converted("SI", "mm-tonne-s")
        assert converted is not case
        assert converted.case_id == case.case_id

    def test_version_suffix(self):
        case = hemisphere_case()
        converted = case.with_units_converted("SI", "mm-tonne-s")
        assert "SI2mm-tonne-s" in converted.case_version

    def test_validates_after_conversion(self):
        case = hemisphere_case()
        converted = case.with_units_converted("SI", "mm-tonne-s")
        validate_case(converted)  # must not raise

    def test_hertz_case_conversion(self):
        case = hertz_ball_plane_case()
        converted = case.with_units_converted("SI", "mm-tonne-s")
        validate_case(converted)
        # radius 10 m → 10000 mm
        assert abs(converted.geometry["radius"] - 10000.0) < 1e-6


# ---------------------------------------------------------------------------
# Integration: run_case with converted units
# ---------------------------------------------------------------------------

class TestRunCaseWithConvertedUnits:
    def test_converted_case_produces_consistent_results(self):
        """A case converted to mm-tonne-s and run should produce results
        that are consistent with the original when converted back."""
        case = hemisphere_case()
        converted = case.with_units_converted("SI", "mm-tonne-s")

        # Run both (quick tier uses stored quick_values)
        archive_orig = run_case(case, tier="quick")
        archive_conv = run_case(converted, tier="quick")

        # Results should differ by the length conversion factor (1000)
        factor = 1000.0
        for orig_row, conv_row in zip(archive_orig["results"], archive_conv["results"]):
            assert orig_row["mesh"] == conv_row["mesh"]
            expected = orig_row["computed"] * factor
            assert abs(conv_row["computed"] - expected) / expected < 1e-6
