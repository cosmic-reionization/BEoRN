"""nonuniform-fix S3: weighted halo meshes and log-M weight-split painting.

See fst_stochastic/docs/src_fix_plan_2026-10-01.md (D5)."""
import sys
import types

if "MAS_library" not in sys.modules:
    sys.modules["MAS_library"] = types.SimpleNamespace(MASL=None)

import numpy as np
import pytest
from types import SimpleNamespace

from beorn.painting.coordinator import PaintingCoordinator
from beorn.structs.halo_catalog import HaloCatalog
from beorn.structs.parameters import Parameters


def real_parameters(numerics="v2"):
    p = Parameters()
    p.solver.numerics = numerics
    p.simulation.Ncell = 16
    p.simulation.Lbox = 20.0
    p.simulation.fft_backend = "numpy"
    p.simulation.minimum_grid_size_heat = 1
    p.simulation.minimum_grid_size_lyal = 1
    p.source.halo_mass_min = 1e9
    p.cosmo_sim.particle_mapping_backend = "numpy"
    return p


def coordinator_for(params):
    """A PaintingCoordinator shell exposing only what the assignment helpers use."""
    c = PaintingCoordinator.__new__(PaintingCoordinator)
    c.parameters = params
    return c


def catalog(params, masses, alphas=None, positions=None, weights=None):
    masses = np.asarray(masses, dtype=float)
    n = masses.size
    if positions is None:
        positions = np.random.default_rng(1).uniform(0, params.simulation.Lbox, size=(n, 3))
    if alphas is None:
        alphas = np.full(n, 0.5)
    return HaloCatalog(positions=positions, masses=masses, parameters=params, alphas=np.asarray(alphas, float),
                       weights=weights)


# ---------------------------------------------------------------------------- weights
def edges_at(z_centres_dex):
    """Bin edges whose geometric centres are 10**z_centres_dex (uniform spacing)."""
    c = np.asarray(z_centres_dex, float)
    d = c[1] - c[0]
    return 10 ** np.concatenate([c - d / 2, [c[-1] + d / 2]])


def test_weights_on_nodes_and_midpoints():
    c = coordinator_for(real_parameters())
    e = edges_at([9.5, 10.0, 10.5, 11.0])
    m = 10 ** np.array([10.0, 10.25, 10.5])
    lo, hi, wl, wh, dropped = c._mass_interpolation_weights(m, e)
    np.testing.assert_allclose(wl + wh + dropped, 1.0)
    assert (lo[0], wl[0], wh[0]) == (1, 1.0, 0.0)            # exactly on node 1
    np.testing.assert_allclose([wl[1], wh[1]], [0.5, 0.5])    # midpoint between nodes 1 and 2
    assert (lo[1], hi[1]) == (1, 2)
    assert np.all(dropped == 0)


def test_weights_clamp_above_last_node():
    c = coordinator_for(real_parameters())
    e = edges_at([9.5, 10.0, 10.5])
    lo, hi, wl, wh, dropped = c._mass_interpolation_weights(np.array([10 ** 10.6]), e)
    assert lo[0] == hi[0] == 2 and wl[0] == 1.0 and wh[0] == 0.0


def test_zero_knot_at_the_cutoff_inside_an_interval():
    """c_j <= M_min < M < c_{j+1}: the weight on j+1 is measured from M_min, -> 0 at the cutoff."""
    c = coordinator_for(real_parameters())
    e = edges_at([8.8, 9.2, 9.6])                               # nodes 8.8 | 9.2 | 9.6, M_min = 9.0
    m = 10 ** np.array([9.0 + 1e-9, 9.1, 9.2])
    lo, hi, wl, wh, dropped = c._mass_interpolation_weights(m, e)
    on_node1 = np.where(hi == 1, wh, 0.0) + np.where(lo == 1, wl, 0.0)
    np.testing.assert_allclose(on_node1, [0.0, 0.5, 1.0], atol=1e-6)   # (logM - 9.0)/(9.2 - 9.0)
    assert np.all(lo[:2] == 0) and np.all(hi[:2] == 1)
    np.testing.assert_allclose(wl + wh + dropped, 1.0)


def test_zero_knot_below_the_first_node():
    """M_min < M < c_0: weight (logM - log M_min)/(log c_0 - log M_min) on node 0, rest dropped."""
    c = coordinator_for(real_parameters())
    e = edges_at([9.2, 9.6, 10.0])
    lo, hi, wl, wh, dropped = c._mass_interpolation_weights(np.array([10 ** 9.1]), e)
    assert lo[0] == 0
    np.testing.assert_allclose([wl[0], wh[0], dropped[0]], [0.5, 0.0, 0.5])


# ---------------------------------------------------------------------------- catalog
def test_weighted_mesh_none_is_counts_and_weights_sum():
    p = real_parameters()
    cat = catalog(p, [2e9, 3e9, 4e9])
    counts = cat.to_mesh()
    assert counts.sum() == 3
    w = np.array([0.2, 0.5, 1.0])
    wcat = cat.with_weights(w)
    np.testing.assert_allclose(wcat.to_mesh().sum(), w.sum(), rtol=1e-6)
    assert wcat.weight_sum == pytest.approx(1.7)
    sub = wcat.at_indices(np.array([1, 2]))
    np.testing.assert_allclose(sub.weights, [0.5, 1.0])


# ---------------------------------------------------------------------------- assignment
def test_legacy_assignment_is_nearest_bin():
    p = real_parameters("legacy")
    c = coordinator_for(p)
    masses = 10 ** np.array([9.1, 9.3, 9.7, 10.2])
    cat = catalog(p, masses)
    mass_range = np.repeat(edges_at([9.25, 9.75, 10.25])[:, None], 1, axis=1)
    solver_cfg = SimpleNamespace(halo_mass_accretion_alpha=np.array([0.4, 0.6]))
    out = list(c._iter_bin_assignments(cat, mass_range, range(1), range(3), solver_cfg))
    assert sum(o[3] for o in out) == cat.size
    for _, mass_index, bin_cat, count, members in out:
        assert bin_cat.weights is None
        assert np.all((cat.masses[members] >= mass_range[mass_index, 0]) & (cat.masses[members] < mass_range[mass_index + 1, 0]))


def test_v2_assignment_conserves_weight_per_halo():
    p = real_parameters("v2")
    c = coordinator_for(p)
    masses = 10 ** np.array([9.05, 9.3, 9.6, 10.1, 10.6])
    cat = catalog(p, masses, alphas=[0.45, 0.45, 0.55, 0.55, 0.45])
    mass_range = np.stack([edges_at([9.25, 9.75, 10.25]), edges_at([9.2, 9.7, 10.2])], axis=1)
    solver_cfg = SimpleNamespace(halo_mass_accretion_alpha=np.array([0.4, 0.5, 0.6]))
    out = list(c._iter_bin_assignments(cat, mass_range, range(2), range(3), solver_cfg))
    per_halo = np.zeros(cat.size)
    for _, _, bin_cat, count, members in out:
        np.add.at(per_halo, members, bin_cat.weights)
        assert count == pytest.approx(bin_cat.weights.sum())
    per_halo += 0.0
    total = per_halo.sum() + c._knot_dropped_weight
    assert total == pytest.approx(cat.size)
    # each halo lands in at most two bins of its own alpha column
    seen = {}
    for alpha_index, mass_index, _, _, members in out:
        for h in members:
            seen.setdefault(h, set()).add((alpha_index, mass_index))
    assert all(len(v) <= 2 for v in seen.values())


def test_split_painting_is_linear_in_the_weights():
    """Painting a halo with weight w equals w x painting it with weight 1 (all three fields),
    so weight-splitting reproduces the linearly interpolated profile."""
    p = real_parameters("v2")
    c = coordinator_for(p)
    cat = catalog(p, [3e9], positions=np.array([[10.0, 10.0, 10.0]]))
    r = np.logspace(-2, 1.5, 60)
    rl = np.logspace(-3, 1.5, 80)
    prof = (np.array(1.2), 1e-12 / (1 + rl) ** 2, 50.0 / (1 + r) ** 2)
    kw = dict(z=10.0, radial_grid=r / 11.0, r_lyal=rl, profiles_of_bin=prof,
              store_grids=("Grid_xHII", "Grid_Temp", "Grid_xal"))
    one = c.paint_single_mass_bin(halo_catalog=cat, **kw)
    w = 0.37
    part = c.paint_single_mass_bin(halo_catalog=cat.with_weights(np.array([w])), **kw)
    for a, b in zip(one, part):
        np.testing.assert_allclose(b, w * a, rtol=1e-5, atol=1e-12 * np.abs(a).max())
