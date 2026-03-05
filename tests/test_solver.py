import numpy as np
from types import SimpleNamespace

import beorn.precomputation.solver as solver_mod
from beorn.precomputation.solver import RadiationProfileFstSolver, RadiationProfileSolver
from beorn.structs.radiation_profiles import RadiationProfilesFStarGrid, RadiationProfiles


def make_parameters():
    return SimpleNamespace(
        cosmology=SimpleNamespace(
            Ob=0.05,
            Om=0.3,
            h=0.7,
            HI_frac=0.76,
            clumping=1.0,
            z_decoupling=135.0,
        ),
        solver=SimpleNamespace(
            fXh="constant",
        ),
        source=SimpleNamespace(
            f_st=0.05,
            f_st_grid_min=0.01,
            f_st_grid_max=0.03,
            f_st_grid_n=3,
            energy_cutoff_min_xray=500.0,
            energy_cutoff_max_xray=2000.0,
            halo_mass_min=1e7,
            Mp=1e8,
            g1=0.49,
            g2=-0.61,
            Mt=7e7,
            g3=4.0,
            g4=-1.0,
            alS_xray=1.5,
            energy_max_sed_xray=2000.0,
            energy_min_sed_xray=500.0,
            xray_normalisation=1e40,
        ),
        simulation=SimpleNamespace(
            halo_mass_bins=np.array([1e8, 1e9, 1e10]),
            halo_mass_bin_centers=np.array([3e8, 3e9]),
            halo_mass_accretion_alpha=np.array([0.5, 1.0]),
            halo_mass_accreation_alpha_bin_centers=np.array([0.75]),
            halo_mass_bin_n=3,
        ),
    )


def test_build_f_st_grid():
    params = make_parameters()
    redshifts = np.array([15.0, 12.0])

    solver = RadiationProfileFstSolver(params, redshifts)
    np.testing.assert_allclose(
        solver.f_st_grid,
        np.array([0.01, 0.02, 0.03]),
    )


def test_build_f_st_grid_invalid_range_raises():
    params = make_parameters()
    params.source.f_st_grid_min = 0.2
    params.source.f_st_grid_max = 0.01
    redshifts = np.array([15.0, 12.0])

    try:
        RadiationProfileFstSolver(params, redshifts)
    except ValueError as exc:
        assert "f_st_grid_min must be smaller than f_st_grid_max" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid f_st range")


def test_parameters_with_f_st_returns_independent_copy():
    params = make_parameters()
    redshifts = np.array([15.0, 12.0])
    solver = RadiationProfileFstSolver(params, redshifts)

    p2 = solver._parameters_with_f_st(0.123)

    assert p2 is not params
    assert p2.source is not params.source
    assert p2.source.f_st == 0.123
    assert params.source.f_st == 0.05


def test_solve_returns_fstar_grid(monkeypatch):
    params = make_parameters()
    redshifts = np.array([15.0, 12.0])
    solver = RadiationProfileFstSolver(params, redshifts)

    halo_mass_bins = np.ones((3, 2, 2))
    halo_mass = np.ones((2, 1, 2))
    halo_mass_derivative = 10.0 * np.ones((2, 1, 2))

    def fake_mass_accretion(parameters, z_bins_in, halo_bins_in, alpha_in):
        # First call in solve(): halo_mass_bins
        if np.shape(halo_bins_in)[0] == 3:
            return halo_mass_bins, None
        # Second call in solve(): halo_mass, halo_mass_derivative
        return halo_mass, halo_mass_derivative

    monkeypatch.setattr(solver_mod, "mass_accretion", fake_mass_accretion)

    def fake_compute_single_f_st_profiles(self, parameters, x_e, halo_mass_in, halo_mass_derivative_in):
        f = parameters.source.f_st
        z_n = len(self.z_bins)
        r_n = len(self.r_grid)
        mass_n = self.parameters.simulation.halo_mass_bin_n - 1
        alpha_n = len(self.parameters.simulation.halo_mass_accretion_alpha) - 1
        rlya_n = 1000

        r_bubble = np.full((mass_n, alpha_n, z_n), f)
        rho_xray = np.full((r_n, mass_n, alpha_n, z_n), 10.0 * f)
        rho_heat = np.full((r_n, mass_n, alpha_n, z_n), 20.0 * f)
        rho_alpha = np.full((rlya_n, mass_n, alpha_n, z_n), 100.0 * f)
        return r_bubble, rho_xray, rho_heat, rho_alpha

    monkeypatch.setattr(
        RadiationProfileFstSolver,
        "_compute_single_f_st_profiles",
        fake_compute_single_f_st_profiles,
    )

    profiles = solver.solve()

    assert isinstance(profiles, RadiationProfilesFStarGrid)

    assert profiles.R_bubble.shape == (2, 1, 3, 2)
    assert profiles.rho_xray.shape == (len(solver.r_grid), 2, 1, 3, 2)
    assert profiles.rho_heat.shape == (len(solver.r_grid), 2, 1, 3, 2)
    assert profiles.rho_alpha.shape == (1000, 2, 1, 3, 2)

    np.testing.assert_allclose(profiles.f_st_grid, np.array([0.01, 0.02, 0.03]))

    for i, f in enumerate(profiles.f_st_grid):
        np.testing.assert_allclose(profiles.R_bubble[:, :, i, :], f)
        np.testing.assert_allclose(profiles.rho_xray[:, :, :, i, :], 10.0 * f)
        np.testing.assert_allclose(profiles.rho_heat[:, :, :, i, :], 20.0 * f)
        np.testing.assert_allclose(profiles.rho_alpha[:, :, :, i, :], 100.0 * f)


def test_compute_single_f_st_profiles_restores_original_parameters(monkeypatch):
    params = make_parameters()
    redshifts = np.array([15.0, 12.0])
    solver = RadiationProfileFstSolver(params, redshifts)

    solver.halo_mass_evolution = np.ones((2, 1, 2))
    solver.halo_mass_derivative = np.ones((2, 1, 2))

    alt_params = solver._parameters_with_f_st(0.02)
    original_parameters = solver.parameters

    def fake_R_bubble(self):
        return np.zeros((2, 1, 2))

    def fake_rho_xray(self, rr, x_e):
        return np.zeros((len(rr), 2, 1, 2))

    def fake_rho_heat(self, rho_xray):
        return np.zeros_like(rho_xray)

    def fake_rho_alpha_profile(parameters, z_bins_in, r_lyal, halo_mass, halo_mass_derivative):
        return np.zeros((len(r_lyal), 2, 1, 2))

    monkeypatch.setattr(RadiationProfileFstSolver, "R_bubble", fake_R_bubble)
    monkeypatch.setattr(RadiationProfileFstSolver, "rho_xray", fake_rho_xray)
    monkeypatch.setattr(RadiationProfileFstSolver, "rho_heat", fake_rho_heat)
    monkeypatch.setattr(solver_mod, "rho_alpha_profile", fake_rho_alpha_profile)

    x_e = np.array([2e-4, 2e-4])

    solver._compute_single_f_st_profiles(
        alt_params,
        x_e,
        solver.halo_mass_evolution,
        solver.halo_mass_derivative,
    )

    assert solver.parameters is original_parameters
    assert solver.parameters.source.f_st == params.source.f_st

def test_legacy_profile_cache_namespace():
    params = make_parameters()
    solver = RadiationProfileSolver(params, np.array([15.0, 12.0]))
    assert solver.profile_cache_namespace() == "radiation_profiles_legacy"


def test_fstar_profile_cache_namespace_includes_grid_definition():
    params = make_parameters()
    solver = RadiationProfileFstSolver(params, np.array([15.0, 12.0]))
    namespace = solver.profile_cache_namespace()
    assert namespace.startswith("radiation_profiles_fstar_grid_")
    assert "fmin_0.01" in namespace
    assert "fmax_0.03" in namespace
    assert "n_3" in namespace


def test_get_or_compute_profiles_legacy_uses_legacy_cache_namespace():
    params = make_parameters()
    solver = RadiationProfileSolver(params, np.array([15.0, 12.0]))
    captured = {}

    class DummyProfiles:
        pass

    class DummyHandler:
        def load_file(self, parameters, cls, **kwargs):
            captured["load_cls"] = cls
            captured["load_kwargs"] = kwargs
            return DummyProfiles()

    profiles = solver.get_or_compute_profiles(DummyHandler())
    assert isinstance(profiles, DummyProfiles)
    assert captured["load_cls"] is RadiationProfiles
    assert captured["load_kwargs"]["cache_namespace"] == "radiation_profiles_legacy"


def test_get_or_compute_profiles_fstar_uses_fstar_cache_namespace():
    params = make_parameters()
    solver = RadiationProfileFstSolver(params, np.array([15.0, 12.0]))
    captured = {}

    class DummyProfiles:
        pass

    class DummyHandler:
        def load_file(self, parameters, cls, **kwargs):
            captured["load_cls"] = cls
            captured["load_kwargs"] = kwargs
            return DummyProfiles()

    profiles = solver.get_or_compute_profiles(DummyHandler())
    assert isinstance(profiles, DummyProfiles)
    assert captured["load_cls"] is RadiationProfilesFStarGrid
    assert captured["load_kwargs"]["cache_namespace"] == solver.profile_cache_namespace()