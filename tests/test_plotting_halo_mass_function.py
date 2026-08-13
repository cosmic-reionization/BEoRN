"""Unit tests for plot_halo_mass_function's analytical-curve overlay, which
now uses BEoRN's own beorn.mass_function.HaloMassFunction instead of the
external `hmf` package (no optional dependency required)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from beorn.structs import Parameters, HaloCatalog
from beorn.mass_function import HaloMassFunction
from beorn.plotting.halo_mass_function import plot_halo_mass_function


@pytest.fixture
def halo_catalog():
    params = Parameters()
    rng = np.random.default_rng(0)
    n = 2000
    masses = np.logspace(9, 13, n)
    positions = rng.uniform(0, params.Lbox_hunits, size=(n, 3))
    return HaloCatalog(positions=positions, masses=masses, parameters=params, redshift=7.0)


def test_plot_halo_mass_function_draws_analytical_curve_without_hmf_package(halo_catalog):
    fig, ax = plt.subplots()
    plot_halo_mass_function(ax, halo_catalog)
    # data-marker line + analytical curve
    assert len(ax.get_lines()) == 2
    plt.close(fig)


def test_plot_halo_mass_function_analytical_false_skips_curve(halo_catalog):
    fig, ax = plt.subplots()
    plot_halo_mass_function(ax, halo_catalog, analytical=False)
    # errorbar's own data markers are a Line2D too -- just the 1, no extra
    # analytical-curve line on top of it.
    assert len(ax.get_lines()) == 1
    plt.close(fig)


def test_plot_halo_mass_function_analytical_curve_matches_halo_mass_function(halo_catalog):
    fig, ax = plt.subplots()
    plot_halo_mass_function(ax, halo_catalog, analytical_model='ST')
    # errorbar's data-marker line is added first; the analytical curve is
    # the last line added.
    line = ax.get_lines()[-1]
    M_grid = line.get_xdata()
    n_plotted = line.get_ydata()

    hmf = HaloMassFunction(halo_catalog.parameters, model='st',
                            delta_c=halo_catalog.parameters.halo_sim.delta_c)
    expected = hmf.dndlnm(M_grid, halo_catalog.redshift)

    np.testing.assert_allclose(n_plotted, expected)
    plt.close(fig)


def test_plot_halo_mass_function_accepts_press_schechter(halo_catalog):
    fig, ax = plt.subplots()
    plot_halo_mass_function(ax, halo_catalog, analytical_model='PS')
    # data-marker line + analytical curve
    assert len(ax.get_lines()) == 2
    plt.close(fig)


def test_plot_halo_mass_function_unknown_model_raises(halo_catalog):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Unknown model"):
        plot_halo_mass_function(ax, halo_catalog, analytical_model='Tinker08')
    plt.close(fig)


def test_plot_halo_mass_function_raises_without_redshift():
    params = Parameters()
    rng = np.random.default_rng(0)
    masses = np.logspace(9, 13, 500)
    positions = rng.uniform(0, params.Lbox_hunits, size=(500, 3))
    catalog = HaloCatalog(positions=positions, masses=masses, parameters=params, redshift=None)

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="redshift is not set"):
        plot_halo_mass_function(ax, catalog)
    plt.close(fig)
