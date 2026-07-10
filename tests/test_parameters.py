import sys
import types

# Optional dependency stub: importing beorn may pull in the pylians backend.
if "MAS_library" not in sys.modules:
    sys.modules["MAS_library"] = types.SimpleNamespace(MASL=None)

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
