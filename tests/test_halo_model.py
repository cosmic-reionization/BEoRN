from pathlib import Path

import numpy as np
import pytest

from beorn.structs import Parameters
from beorn.halo_model import (
    FIELDS,
    FunctionTracer,
    HaloModel,
    HaloModel21cm,
    HaloModelDM,
    MatterTracer,
    ProfileTracer,
    TopHatTracer,
    Tracer,
    halo_bias,
    nfw_ft,
    profile_ft,
    tophat_ft,
)


@pytest.fixture(scope="module")
def param():
    p = Parameters()
    p.solver.redshifts = np.arange(20, 5.5, -0.5)
    p.solver.halo_mass_nbin = 30
    p.solver.halo_mass_accretion_alpha = np.array([0.7, 0.9])
    return p


@pytest.fixture(scope="module")
def profiles(param):
    from beorn.precomputation import RadiationProfileSolver
    return RadiationProfileSolver(param, param.solver.redshifts).solve()


# ── profile transforms ───────────────────────────────────────────────

def test_profile_ft_gaussian_on_coarse_log_grid():
    sigma = 2.0
    r = np.logspace(-2, np.log10(600), 200)
    rho = np.exp(-r ** 2 / (2 * sigma ** 2))
    k = np.logspace(-3, 1.3, 40)
    W = profile_ft(k, r, rho[:, None])[:, 0]
    W_true = (2 * np.pi * sigma ** 2) ** 1.5 * np.exp(-k ** 2 * sigma ** 2 / 2)
    assert np.max(np.abs(W - W_true)) / W_true[0] < 2e-3


def test_profile_ft_power_law_matches_fine_quadrature():
    r = np.logspace(-2, np.log10(600), 200)
    rho = np.where(r < 300, r ** -2.0, 0.0)
    r_fine = np.logspace(-2, np.log10(600), 400_000)
    rho_fine = np.where(r_fine < 300, r_fine ** -2.0, 0.0)
    for k in (0.01, 1.0, 10.0):
        ref = np.trapezoid(4 * np.pi * r_fine ** 2 * rho_fine * np.sinc(k * r_fine / np.pi), r_fine)
        ref += 4 * np.pi / 3 * r[0] ** 3 * rho[0]
        W = profile_ft(np.array([k]), r, rho[:, None])[0, 0]
        assert W == pytest.approx(ref, rel=2e-3)


def test_tophat_and_nfw_limits():
    k = np.array([1e-6])
    R = np.array([0.5, 2.0])
    np.testing.assert_allclose(tophat_ft(k, R)[0], 4 * np.pi / 3 * R ** 3, rtol=1e-8)
    np.testing.assert_allclose(nfw_ft(np.array([1e-5]), [0.01, 0.1], [5.0, 10.0]), 1.0, rtol=1e-4)


def test_bias_models_large_mass_more_biased():
    sigma = np.array([3.0, 1.0, 0.3])
    for model in ("press_schechter", "sheth_tormen", "tinker10"):
        b = halo_bias(sigma, model)
        assert np.all(np.diff(b) > 0)


# ── matter ───────────────────────────────────────────────────────────

def test_matter_two_halo_recovers_linear_on_large_scales(param):
    hm = HaloModelDM(param)
    k = np.logspace(-3, 1, 12)
    out = hm.power_spectrum(k, z=0.0)
    np.testing.assert_allclose(out['P_2h'][:3] / out['P_lin'][:3], 1.0, rtol=1e-3)
    assert out['P'][-1] > 5 * out['P_lin'][-1]   # 1h boost on small scales


def test_bispectrum_is_reserved(param):
    with pytest.raises(NotImplementedError):
        HaloModel(param).bispectrum()


# ── generic tracers ──────────────────────────────────────────────────

def test_matter_tracer_registered_by_default(param):
    hm = HaloModel(param)
    assert isinstance(hm.tracers['m'], MatterTracer)
    assert 'm' not in HaloModel(param, include_matter=False).tracers


def test_function_tracer_point_sources(param):
    """Point-like tracer with W = N(M): P^1h = sum n N^2 dlnM, P^2h = (sum n b N dlnM)^2 P_lin."""
    edges = np.geomspace(1e9, 1e13, 41)

    def N(k, M, z):
        return np.ones((len(k), 1)) * (M / 1e11)[None, :]

    hm = HaloModel(param, tracers=[FunctionTracer('g', N, M_edges=edges)])
    k = np.logspace(-2, 0, 5)
    out = hm.power_spectra(k, 2.0, tracers=['g'])
    M, dlnM = hm.grid_for(2.0, 'g')
    w, b = hm.dndlnm(M, 2.0) * dlnM, hm.bias(M, 2.0)
    np.testing.assert_allclose(out['P_g_g_1h'], np.sum(w * (M / 1e11) ** 2), rtol=1e-12)
    np.testing.assert_allclose(out['P_g_g_2h'], np.sum(w * b * M / 1e11) ** 2 * hm.plin(k, 2.0), rtol=1e-12)
    assert out['mean_g'] == pytest.approx(np.sum(w * M / 1e11), rel=1e-12)


def test_function_tracer_custom_bias(param):
    edges = np.geomspace(1e10, 1e12, 11)
    one = FunctionTracer('g', lambda k, M, z: np.ones((len(k), len(M))), M_edges=edges,
                         bias=lambda M, z: 2.0 * np.ones_like(M))
    hm = HaloModel(param, tracers=[one], include_matter=False)
    k = np.array([0.01])
    M, dlnM = hm.grid_for(1.0, 'g')
    n = np.sum(hm.dndlnm(M, 1.0) * dlnM)
    assert hm.power_spectrum(k, 1.0, 'g')['P_2h'][0] == pytest.approx(4 * n ** 2 * hm.plin(k, 1.0)[0])


def test_binned_tracers_map_masses_to_bins(param):
    edges = np.geomspace(1e9, 1e12, 4)
    R = np.array([0.5, 1.0, 2.0])
    r = np.logspace(-2, 1.5, 300)
    rho = np.where(r[:, None] < R[None, :], 1.0, 0.0)
    top = TopHatTracer('t', R, edges)
    prof = ProfileTracer('p', r, rho, edges)
    hm = HaloModel(param, tracers=[top, prof])
    k = np.array([1e-3])
    M = np.array([1e8, 2e9, 5e10, 5e11, 1e13])
    W = top.window(k, M, 0.0)[0]
    np.testing.assert_allclose(W, [0, *(4 * np.pi / 3 * R ** 3), 0], rtol=1e-6)
    np.testing.assert_allclose(prof.window(k, M, 0.0)[0], W, rtol=5e-2)   # sampled step edge
    out = hm.power_spectra(np.logspace(-2, 0, 4), 0.0)
    assert {'P_m_t', 'P_t_p', 'P_p_p', 'mean_t'} <= set(out)


def test_custom_tracer_subclass(param):
    class Constant(Tracer):
        def window(self, k, M, z):
            return np.full((len(k), len(M)), 3.0)

    hm = HaloModel(param, tracers=[Constant('c')])
    out = hm.power_spectrum(np.array([0.1]), 0.0, 'c', 'm')
    assert np.isfinite(out['P'][0]) and out['P'][0] > 0


def test_power_spectrum_matches_power_spectra(param):
    hm = HaloModelDM(param)
    k = np.logspace(-2, 1, 6)
    np.testing.assert_allclose(hm.power_spectrum(k, 1.0)['P'], hm.power_spectra(k, 1.0)['P_m_m'])


# ── 21 cm ────────────────────────────────────────────────────────────

def test_21cm_global_history_and_spectra(param, profiles):
    hm = HaloModel21cm(param, profiles)
    k = np.logspace(-2, 1, 10)
    z = [16.0, 12.0, 9.0, 7.0]
    out = hm.compute(k, z)

    assert out['P_21'].shape == (len(z), k.size)
    assert np.all(np.isfinite(out['P_21'])) and np.all(out['P_21'] > 0)
    assert np.all(np.diff(out['x_HII']) > 0)          # reionisation proceeds
    assert np.all(np.diff(out['T_k'][1:]) > 0)        # X-ray heating after the trough
    assert np.all(out['x_HII'] < 1)

    # P_21 is the A_X A_Y-weighted sum of the field spectra
    iz = 2
    P = sum((1 if X == Y else 2) * out['A'][X][iz] * out['A'][Y][iz] * out[f'P_{X}_{Y}'][iz]
            for i, X in enumerate(FIELDS) for Y in FIELDS[i:])
    np.testing.assert_allclose(out['P_21'][iz], P, rtol=1e-10)


def test_21cm_mean_fields_match_profile_volume_integrals(param, profiles):
    hm = HaloModel21cm(param, profiles)
    z = hm.profile_redshift(9.0)
    M, dlnM = hm.grid_for(z, 'ion')
    R_b = np.asarray(profiles.R_bubble[:, hm.alpha_index, hm._z_index(z)])
    x_HII = np.sum(hm.dndlnm(M, z) * dlnM * 4 * np.pi / 3 * R_b ** 3)
    assert hm.mean_fields(z)['x_HII'] == pytest.approx(x_HII, rel=1e-6)


def test_21cm_bubble_correction_suppresses_ionisation_power(param, profiles):
    k = np.logspace(-1, 1, 5)
    plain = HaloModel21cm(param, profiles, bubble_correction=False).compute(k, 7.0)
    corr = HaloModel21cm(param, profiles, bubble_correction=True, b_1h=1.2).compute(k, 7.0)
    assert np.all(corr['P_ion_ion_1h'] < plain['P_ion_ion_1h'])


def test_21cm_power_vanishes_as_reionisation_completes():
    """With the default overlap correction P_21 peaks mid-reionisation and drops to ~0 at x_HII -> 1."""
    from beorn.precomputation import RadiationProfileSolver
    p = Parameters()
    p.solver.redshifts = np.arange(20, 4.75, -0.25)
    p.solver.halo_mass_nbin = 40
    p.solver.halo_mass_accretion_alpha = np.array([0.7, 0.9])
    hm = HaloModel21cm(p, RadiationProfileSolver(p, p.solver.redshifts).solve())
    out = hm.compute(np.array([0.1, 0.5]), np.arange(9, 4.99, -0.25))
    assert out['x_HII'][-1] == pytest.approx(1.0)
    D2 = out['Delta2_21']
    assert np.all(D2[-1] < 1e-6 * D2.max(axis=0))


# ── profile caching / repeated runs ─────────────────────────────────

def test_profiles_not_cached_by_default(param, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hm = HaloModel21cm(param)
    assert hm.profiles._file_path is None
    assert list(tmp_path.iterdir()) == []


def test_profiles_cached_when_requested(param, tmp_path):
    cache = tmp_path / "cache"
    hm = HaloModel21cm(param, cache_profiles=cache)
    files = list(cache.rglob("RadiationProfiles_*.h5"))
    assert len(files) == 1
    again = HaloModel21cm(param, cache_profiles=cache)          # loaded, not recomputed
    assert Path(again.profiles._file_path) == files[0]
    k = np.logspace(-1, 0, 3)
    np.testing.assert_allclose(again.compute(k, 9.0)['P_21'], hm.compute(k, 9.0)['P_21'], rtol=1e-10)


def test_update_parameters_recomputes_profiles_only(param):
    import copy
    hm = HaloModel21cm(param)
    k = np.logspace(-1, 0, 3)
    ref = hm.compute(k, 8.0)
    linear_power = hm.linear_power

    new = copy.deepcopy(param)
    new.source.Nion = 2 * param.source.Nion
    hm.update_parameters(new)
    out = hm.compute(k, 8.0)
    assert hm.linear_power is linear_power
    assert out['x_HII'][0] > ref['x_HII'][0]

    bad = copy.deepcopy(param)
    bad.cosmology.sigma_8 = 0.7
    with pytest.raises(ValueError, match="cosmology"):
        hm.update_parameters(bad)


def test_matter_matches_pyhalomodel(param):
    """Same sigma(M), P_lin, Sheth-Tormen n/b, NFW + Duffy08: the integrals must agree."""
    halo = pytest.importorskip("pyhalomodel.pyhalomodel")
    from beorn.halo_model import concentration_duffy08
    hdm = HaloModelDM(param)
    c = param.cosmology
    k = np.logspace(-3, 1.5, 10)
    for z in (0.0, 8.0):
        M = hdm.M_grid
        sigmaM = np.sqrt(hdm.hmf.sigma2(M / c.h0, z))
        model = halo.model(z, c.Om, name='Sheth & Tormen (1999)', Dv=200.0, dc=hdm.delta_c)
        prof = halo.matter_profile(k, M, model.virial_radius(M), concentration_duffy08(M, z, c.h0), c.Om)
        _, _, P = model.power_spectrum(k, hdm.plin(k, z), M, sigmaM, {'m': prof}, verbose=False)
        np.testing.assert_allclose(hdm.power_spectrum(k, z)['P'], P['m-m'], rtol=2e-3)
