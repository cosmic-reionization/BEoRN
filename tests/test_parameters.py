import sys
import types

# Optional dependency stub: importing beorn may pull in the pylians backend.
if "MAS_library" not in sys.modules:
    sys.modules["MAS_library"] = types.SimpleNamespace(MASL=None)

import pytest

from beorn.structs.parameters import Parameters


def _make_params(**source_overrides):
    params = Parameters()
    for key, value in source_overrides.items():
        setattr(params.source, key, value)
    return params


def test_profiles_fstar_hash_stable_across_sigma0_sigma1_mpiv_changes():
    """The expensive (mass, alpha, f_st, z) profile cube must stay shared across runs
    that only differ in how the f_st scatter width is sampled -- sigma0/sigma1/mpiv
    only affect painting, not the profile shapes themselves."""
    base = _make_params()
    mass_dependent = _make_params(
        f_st_paint_sigma0=0.22, f_st_paint_sigma1=-0.04, f_st_paint_sigma_mpiv=1e11,
    )
    other_mass_dependent = _make_params(
        f_st_paint_sigma0=0.5, f_st_paint_sigma1=0.3, f_st_paint_sigma_mpiv=5e10,
    )

    assert base.profiles_fstar_hash() == mass_dependent.profiles_fstar_hash()
    assert base.profiles_fstar_hash() == other_mass_dependent.profiles_fstar_hash()


@pytest.mark.parametrize("key, value", [("ode_rtol", 1e-6), ("ode_atol", 1e-10), ("ode_method", "LSODA")])
def test_profile_hashes_change_with_ode_solver_settings(key, value):
    """The tolerances and method change R_bubble and rho_heat, so a cube solved at other
    settings must not be reused (review_2026-09-14 finding 4)."""
    base = _make_params()
    changed = _make_params()
    setattr(changed.solver, key, value)
    assert base.profiles_hash() != changed.profiles_hash()
    assert base.profiles_fstar_hash() != changed.profiles_fstar_hash()


def test_beorn_hash_changes_with_sigma0():
    base = _make_params()
    changed = _make_params(f_st_paint_sigma0=0.22)
    assert base.beorn_hash() != changed.beorn_hash()


def test_beorn_hash_changes_with_sigma1():
    base = _make_params(f_st_paint_sigma0=0.22, f_st_paint_sigma1=0.0)
    changed = _make_params(f_st_paint_sigma0=0.22, f_st_paint_sigma1=-0.04)
    assert base.beorn_hash() != changed.beorn_hash()


def test_beorn_hash_changes_with_sigma_mpiv():
    base = _make_params(f_st_paint_sigma0=0.22, f_st_paint_sigma1=-0.04, f_st_paint_sigma_mpiv=1e11)
    changed = _make_params(f_st_paint_sigma0=0.22, f_st_paint_sigma1=-0.04, f_st_paint_sigma_mpiv=5e10)
    assert base.beorn_hash() != changed.beorn_hash()


def test_beorn_hash_stable_when_parameters_unchanged():
    a = _make_params(f_st_paint_sigma0=0.22, f_st_paint_sigma1=-0.04)
    b = _make_params(f_st_paint_sigma0=0.22, f_st_paint_sigma1=-0.04)
    assert a.beorn_hash() == b.beorn_hash()


# ---------------------------------------------------------------------------
# nonuniform-fix (2026-10): numerics = 'v2' (default) | 'legacy'
# ---------------------------------------------------------------------------

# Hashes of a default Parameters() computed with the code *before* the v2 fields existed
# (outer repo thesan-1 @ 66618b2). A legacy configuration must keep producing them.
_PRE_V2_DEFAULT_HASHES = {
    "profiles_hash": "37859c64",
    "profiles_fstar_hash": "72050e75",
    "beorn_hash": "4fe6dd00",
    "unique_hash": "ca08bfc8ab60f7a1b5876ff82194d491",
}


def test_legacy_numerics_reproduces_pre_v2_hashes():
    p = Parameters()
    p.solver.numerics = "legacy"
    for name, expected in _PRE_V2_DEFAULT_HASHES.items():
        assert getattr(p, name)() == expected, name


def test_legacy_hashes_ignore_v2_only_fields():
    p = Parameters()
    p.solver.numerics = "legacy"
    p.solver.onset_internal_refine = 3
    p.solver.xray_dz_prime = 0.5
    p.solver.bubble_substeps = 50
    p.solver.mass_top_check = False
    p.simulation.mass_interpolation = "none"
    for name, expected in _PRE_V2_DEFAULT_HASHES.items():
        assert getattr(p, name)() == expected, name


def test_v2_is_default_and_changes_hashes():
    p = Parameters()
    assert p.solver.numerics == "v2"
    assert p.mass_interpolation_active
    for name, legacy_hash in _PRE_V2_DEFAULT_HASHES.items():
        assert getattr(p, name)() != legacy_hash, name


@pytest.mark.parametrize("key, value", [
    ("onset_internal_refine", 8),
    ("xray_dz_prime", 0.1),
    ("bubble_substeps", 500),
])
def test_v2_profile_hashes_track_profile_settings(key, value):
    p = Parameters()
    before = (p.profiles_hash(), p.profiles_fstar_hash())
    setattr(p.solver, key, value)
    assert p.profiles_hash() != before[0]
    assert p.profiles_fstar_hash() != before[1]


def test_mass_interpolation_is_a_paint_option_only():
    p = Parameters()
    before = (p.profiles_hash(), p.beorn_hash())
    p.simulation.mass_interpolation = "none"
    assert p.profiles_hash() == before[0]
    assert p.beorn_hash() != before[1]
    assert not p.mass_interpolation_active


def test_legacy_disables_mass_interpolation():
    p = Parameters()
    p.solver.numerics = "legacy"
    assert not p.mass_interpolation_active


@pytest.mark.parametrize("kwargs", [
    {"numerics": "old"},
    {"onset_internal_refine": 0},
    {"bubble_substeps": 5},
    {"xray_dz_prime": 0.0},
])
def test_v2_solver_option_validation(kwargs):
    from beorn.structs.parameters import SolverParameters
    with pytest.raises(ValueError):
        SolverParameters(**kwargs)
