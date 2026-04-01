import numpy as np
import pytest
from types import SimpleNamespace

from beorn.structs.radiation_profiles import RadiationProfilesFStarGrid


def dummy_parameters():
    # BaseStruct only needs something stored in `parameters`
    return SimpleNamespace()


def test_radiation_profiles_fstar_grid_profiles_of_halo_bin_scalar_indices():
    params = dummy_parameters()

    z_n = 4
    m_n = 3
    a_n = 2
    f_n = 5
    r_n = 7
    rlya_n = 11

    z_history = np.linspace(20, 10, z_n)
    halo_mass_bins = np.ones((m_n + 1, a_n + 1, z_n))
    f_st_grid = np.linspace(0.01, 0.2, f_n)

    rho_xray = np.zeros((r_n, m_n, a_n, f_n, z_n))
    rho_heat = np.zeros((r_n, m_n, a_n, f_n, z_n))
    rho_alpha = np.zeros((rlya_n, m_n, a_n, f_n, z_n))
    R_bubble = np.zeros((m_n, a_n, f_n, z_n))

    # Make one unique location easy to identify
    R_bubble[2, 1, 3, 0] = 42.0
    rho_alpha[:, 2, 1, 3, 0] = np.arange(rlya_n) + 100.0
    rho_heat[:, 2, 1, 3, 0] = np.arange(r_n) + 200.0

    r_lyal = np.logspace(-5, 2, rlya_n)
    r_grid_cell = np.logspace(-2, 2, r_n)

    profiles = RadiationProfilesFStarGrid(
        parameters=params,
        z_history=z_history,
        halo_mass_bins=halo_mass_bins,
        f_st_grid=f_st_grid,
        rho_xray=rho_xray,
        rho_heat=rho_heat,
        rho_alpha=rho_alpha,
        R_bubble=R_bubble,
        r_lyal=r_lyal,
        r_grid_cell=r_grid_cell,
    )

    rb, ra, rh = profiles.profiles_of_halo_bin(
        z_index=0,
        alpha_index=1,
        mass_index=2,
        f_st_index=3,
    )

    assert np.isscalar(rb) or rb.shape == ()
    assert ra.shape == (rlya_n,)
    assert rh.shape == (r_n,)

    assert rb == 42.0
    np.testing.assert_allclose(ra, np.arange(rlya_n) + 100.0)
    np.testing.assert_allclose(rh, np.arange(r_n) + 200.0)


def test_radiation_profiles_fstar_grid_profiles_of_halo_bin_slice_indices():
    params = dummy_parameters()

    z_n = 2
    m_n = 4
    a_n = 3
    f_n = 6
    r_n = 5
    rlya_n = 8

    z_history = np.array([15.0, 12.0])
    halo_mass_bins = np.ones((m_n + 1, a_n + 1, z_n))
    f_st_grid = np.linspace(0.01, 0.2, f_n)

    rho_xray = np.ones((r_n, m_n, a_n, f_n, z_n))
    rho_heat = np.arange(r_n * m_n * a_n * f_n * z_n, dtype=float).reshape(r_n, m_n, a_n, f_n, z_n)
    rho_alpha = np.arange(rlya_n * m_n * a_n * f_n * z_n, dtype=float).reshape(rlya_n, m_n, a_n, f_n, z_n)
    R_bubble = np.arange(m_n * a_n * f_n * z_n, dtype=float).reshape(m_n, a_n, f_n, z_n)

    r_lyal = np.logspace(-5, 2, rlya_n)
    r_grid_cell = np.logspace(-2, 2, r_n)

    profiles = RadiationProfilesFStarGrid(
        parameters=params,
        z_history=z_history,
        halo_mass_bins=halo_mass_bins,
        f_st_grid=f_st_grid,
        rho_xray=rho_xray,
        rho_heat=rho_heat,
        rho_alpha=rho_alpha,
        R_bubble=R_bubble,
        r_lyal=r_lyal,
        r_grid_cell=r_grid_cell,
    )

    rb, ra, rh = profiles.profiles_of_halo_bin(
        z_index=1,
        alpha_index=slice(1, 3),
        mass_index=slice(0, 2),
        f_st_index=slice(2, 5),
    )

    assert rb.shape == (2, 2, 3)
    assert ra.shape == (rlya_n, 2, 2, 3)
    assert rh.shape == (r_n, 2, 2, 3)

    np.testing.assert_allclose(
        rb,
        R_bubble[0:2, 1:3, 2:5, 1]
    )
    np.testing.assert_allclose(
        ra,
        rho_alpha[:, 0:2, 1:3, 2:5, 1]
    )
    np.testing.assert_allclose(
        rh,
        rho_heat[:, 0:2, 1:3, 2:5, 1]
    )


def test_radiation_profiles_fstar_grid_requires_1d_f_st_grid():
    params = dummy_parameters()

    z_history = np.array([15.0, 12.0])
    halo_mass_bins = np.ones((2, 2, 2))
    f_st_grid = np.ones((2, 2))   # wrong ndim
    rho_xray = np.ones((3, 1, 1, 2, 2))
    rho_heat = np.ones((3, 1, 1, 2, 2))
    rho_alpha = np.ones((4, 1, 1, 2, 2))
    R_bubble = np.ones((1, 1, 2, 2))
    r_lyal = np.ones(4)
    r_grid_cell = np.ones(3)

    with pytest.raises(AssertionError, match="f_st_grid must be a 1D array"):
        RadiationProfilesFStarGrid(
            parameters=params,
            z_history=z_history,
            halo_mass_bins=halo_mass_bins,
            f_st_grid=f_st_grid,
            rho_xray=rho_xray,
            rho_heat=rho_heat,
            rho_alpha=rho_alpha,
            R_bubble=R_bubble,
            r_lyal=r_lyal,
            r_grid_cell=r_grid_cell,
        )


def test_radiation_profiles_fstar_grid_rejects_nonfinite_values():
    params = dummy_parameters()

    z_history = np.array([15.0, 12.0])
    halo_mass_bins = np.ones((2, 2, 2))
    f_st_grid = np.array([0.01, 0.02])
    rho_xray = np.ones((3, 1, 1, 2, 2))
    rho_heat = np.ones((3, 1, 1, 2, 2))
    rho_alpha = np.ones((4, 1, 1, 2, 2))
    R_bubble = np.ones((1, 1, 2, 2))
    R_bubble[0, 0, 0, 0] = np.nan
    r_lyal = np.ones(4)
    r_grid_cell = np.ones(3)

    with pytest.raises(AssertionError, match="R_bubble contains invalid values"):
        RadiationProfilesFStarGrid(
            parameters=params,
            z_history=z_history,
            halo_mass_bins=halo_mass_bins,
            f_st_grid=f_st_grid,
            rho_xray=rho_xray,
            rho_heat=rho_heat,
            rho_alpha=rho_alpha,
            R_bubble=R_bubble,
            r_lyal=r_lyal,
            r_grid_cell=r_grid_cell,
        )