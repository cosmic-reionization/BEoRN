import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.cosmo.ps_linear import (
    get_power_spectrum, EisensteinHu, CAMBPowerSpectrum, BoltzmannSolver, _sigma8_of_table,
)


@pytest.fixture(scope="module")
def param():
    return Parameters()


@pytest.mark.parametrize("method", [None, "eh", "EH", "eisenstein_hu"])
def test_default_is_eisenstein_hu(param, method):
    ps = get_power_spectrum(method, param)
    assert isinstance(ps, EisensteinHu) and not ps.wiggle


def test_unknown_method_raises(param):
    with pytest.raises(ValueError):
        get_power_spectrum("halofit", param)


def test_lpt_alias_module_reexports_same_objects():
    from beorn.lpt import linear_power
    from beorn.cosmo import ps_linear
    assert linear_power.get_power_spectrum is ps_linear.get_power_spectrum
    assert linear_power.EisensteinHu is ps_linear.EisensteinHu


def test_boltzmann_file_loader_roundtrip(param, tmp_path):
    eh = get_power_spectrum(None, param)
    k = np.logspace(-3, 1, 200)
    path = tmp_path / "pk.dat"
    np.savetxt(path, np.c_[k, eh.P(k)])
    ps = get_power_spectrum("boltzmann", param, ps_file=str(path))
    assert isinstance(ps, BoltzmannSolver)
    kt = np.logspace(-2.5, 0.5, 20)
    np.testing.assert_allclose(ps.P(kt, z=6.0), eh.P(kt, z=6.0), rtol=1e-3)
    # outside the table: E&H continuation, matched at the edge
    np.testing.assert_allclose(ps.P(np.array([1e2])), eh.P(np.array([1e2])), rtol=1e-3)


def test_hmf_accepts_power_spectrum_instance(param):
    from beorn.mass_function import HaloMassFunction
    ps = get_power_spectrum(None, param)
    hmf = HaloMassFunction(param, power_spectrum=ps)
    assert hmf.power_spectrum is ps


camb = pytest.importorskip("camb", reason="CAMB not installed")


@pytest.mark.parametrize("method", ["camb", "CAMB"])
def test_camb_sigma8_and_shape(param, method):
    ps = get_power_spectrum(method, param, kmax=20.0, nk=256)
    assert isinstance(ps, CAMBPowerSpectrum)
    k = np.logspace(-4, 3, 4000)
    assert _sigma8_of_table(k, ps.P(k)) == pytest.approx(param.cosmology.sigma_8, rel=1e-3)
    eh = get_power_spectrum(None, param)
    kt = np.logspace(-3, 0.5, 30)
    ratio = ps.P(kt, z=8.0) / eh.P(kt, z=8.0)
    assert np.all(np.abs(ratio - 1) < 0.15)   # BAO wiggles vs no-wiggle E&H


def test_camb_ps_var_forwarded(param):
    k = np.logspace(-3, 0, 10)
    ref = get_power_spectrum("camb", param, kmax=5.0, normalise_sigma8=False)
    mod = get_power_spectrum("camb", param, kmax=5.0, normalise_sigma8=False, As=4.2e-9)
    np.testing.assert_allclose(mod.P(k) / ref.P(k), 2.0, rtol=1e-3)


def test_class_missing_gives_clear_error(param):
    try:
        import classy  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="classy"):
            get_power_spectrum("class", param)
    else:
        ps = get_power_spectrum("class", param, kmax=5.0)
        k = np.logspace(-4, 3, 4000)
        assert _sigma8_of_table(k, ps.P(k)) == pytest.approx(param.cosmology.sigma_8, rel=1e-3)
