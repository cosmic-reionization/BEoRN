"""Finding 3: profile_to_3Dkernel must centre r=0 on index nGrid//2.

The kernel is ifftshift-ed before the forward FFT (helpers.fourier_multiply_kernel /
precompute_fft), so index nGrid//2 must be r=0 and the cell spacing must be L/N. The
old linspace(-L/2, L/2, N) put r=0 between cells (spacing L/(N-1)), mis-registering the
profile by ~half a cell: a top-hat of radius 1.2 cells occupied the asymmetric block
{-1,0} per axis instead of the symmetric {-1,0,+1}.
"""
import numpy as np

from beorn.painting.helpers import (
    profile_to_3Dkernel,
    _resample_centered,
    stacked_lyal_kernel,
)


def _tophat(R_cells):
    return lambda r: (np.asarray(r, dtype=float) < R_cells).astype(float)


def test_profile_to_3Dkernel_tophat_is_symmetric_about_centre():
    nGrid = 8
    LB = float(nGrid)          # cell size = LB / nGrid = 1
    R = 1.2                     # cells
    kern = profile_to_3Dkernel(_tophat(R), nGrid, LB)
    c = nGrid // 2

    # Along each axis through the centre, occupancy must be the symmetric {c-1, c, c+1}.
    for axis in range(3):
        idx = [c, c, c]
        line = np.empty(nGrid)
        for k in range(nGrid):
            idx[axis] = k
            line[k] = kern[idx[0], idx[1], idx[2]]
        occ = set(np.where(line > 0)[0].tolist())
        assert occ == {c - 1, c, c + 1}, f"axis {axis}: occupied {sorted(occ)}, want {{{c-1},{c},{c+1}}}"

    # The full 3D occupancy set must be symmetric under negation about the centre.
    occ = set(map(tuple, np.argwhere(kern > 0) - c))
    assert occ == {tuple(-np.array(o)) for o in occ}, "kernel occupancy is not centro-symmetric"


def test_profile_to_3Dkernel_peak_at_centre_index():
    nGrid = 8
    LB = float(nGrid)
    # A monotonically decreasing profile peaks at r=0, which must land on index nGrid//2.
    kern = profile_to_3Dkernel(lambda r: np.exp(-np.asarray(r, float)), nGrid, LB)
    peak = np.unravel_index(np.argmax(kern), kern.shape)
    assert peak == (nGrid // 2, nGrid // 2, nGrid // 2)


def test_resample_centered_is_3d_and_centro_symmetric():
    # A centred coarse kernel must resample to a full 3D fine kernel (not a broadcast
    # diagonal) that stays centro-symmetric about the fine centre (finding 3).
    nGrid_min, nGrid = 8, 16
    coarse = profile_to_3Dkernel(lambda r: np.exp(-np.asarray(r, float)), nGrid_min, float(nGrid_min))
    fine = _resample_centered(coarse, nGrid)
    assert fine.shape == (nGrid, nGrid, nGrid)
    c = nGrid // 2
    # the centre cell carries the max value (ties are possible at the c-index seam)
    assert fine[c, c, c] == fine.max()
    # centro-symmetric in the inner region (the outermost -N/2 Nyquist cell has no
    # +N/2 mirror on an even grid, which is the FFT convention, not an asymmetry bug).
    for o in range(1, c - 1):
        assert np.allclose(fine[c + o], fine[c - o]), f"asymmetric at offset {o}"
    # NOT the old bug (a 1D diagonal broadcast over the last axis would make every
    # slice along an axis identical): here the value genuinely varies plane to plane.
    assert not np.allclose(fine[c], fine[c - 2])


def test_stacked_lyal_kernel_shape_and_centre():
    nGrid, nGrid_min = 16, 8
    LBox = 4.0
    rr = np.linspace(0, 20, 400)
    lyal = np.exp(-rr)          # decays; hits ~0 well inside the range so ind_lya_0 exists
    lyal[rr > 8] = 0.0
    kern = stacked_lyal_kernel(rr, lyal, LBox, nGrid, nGrid_min)
    assert kern.shape == (nGrid, nGrid, nGrid)
    # central kernel dominates -> peak at the fine centre
    assert np.unravel_index(np.argmax(kern), kern.shape) == (nGrid // 2,) * 3


# ---------------------------------------------------------------------------
# review_2026-09-14 finding 1: the source's own cell must carry the cell-averaged profile.
#
# The production heating / Lyman-alpha profiles are tabulated from rr[0] > 0 and rise like
# 1/r^2. With r=0 on the kernel origin (the finding-3 convention above), the stacked builders'
# fill_value=0 interpolators returned exactly 0 there. The test above uses rr starting at 0,
# which is why it never caught this.
# ---------------------------------------------------------------------------
import pytest
from scipy.interpolate import interp1d

from beorn.painting import helpers as _helpers
from beorn.painting.helpers import stacked_T_kernel


def _steep_profile(r_max=5.0):
    rr = np.logspace(-2, np.log10(20.0), 400)   # first radius strictly > 0, like production
    vals = rr ** -2.0
    vals[rr > r_max] = 0.0
    return rr, vals


@pytest.mark.parametrize("n_sub", [2, 8, 64])
def test_centre_cell_average_of_constant_profile_is_exact(n_sub):
    rr = np.logspace(-2, 1, 50)
    vals = np.full(rr.size, 3.5)
    assert _helpers._centre_cell_average(rr, vals, cell=0.5, n_sub=n_sub) == pytest.approx(3.5, rel=1e-12)


@pytest.mark.parametrize("n_sub", [8, 64])
def test_centre_cell_average_of_r_squared_matches_midpoint_rule_exactly(n_sub):
    # mean of r^2 over a cube of side a is a^2/4; the midpoint rule on n points per axis gives
    # a^2/4 * (1 - 1/n^2). Checks the octant symmetry and the cell scaling.
    a = 0.7
    rr = np.linspace(0.0, 1.0, 400001)
    got = _helpers._centre_cell_average(rr, rr ** 2, cell=a, n_sub=n_sub)
    assert got == pytest.approx(a ** 2 / 4 * (1 - 1 / n_sub ** 2), rel=1e-8)


@pytest.mark.parametrize("r_min", [1e-2, 1e-5])   # heating-like and Lyman-alpha-like floors
def test_centre_cell_average_default_resolution_is_converged(r_min):
    # Production cell size. An inverse-square profile tabulated down to 1e-5 converges only as
    # 1/n_sub under the midpoint rule, so require the default to sit within 1.5% of a 256^3
    # reference, and to under- rather than over-estimate the singular core.
    cell = 64.6917 / 128
    rr = np.logspace(np.log10(r_min), np.log10(20.0), 2000)
    vals = rr ** -2.0
    default = _helpers._centre_cell_average(rr, vals, cell)
    reference = _helpers._centre_cell_average(rr, vals, cell, n_sub=256)
    assert default == pytest.approx(reference, rel=0.015)
    assert default <= reference * (1 + 1e-3)


def test_centre_cell_average_rejects_odd_subsampling():
    rr, vals = _steep_profile()
    with pytest.raises(ValueError):
        _helpers._centre_cell_average(rr, vals, cell=1.0, n_sub=5)


@pytest.mark.parametrize("builder", [stacked_T_kernel, stacked_lyal_kernel])
@pytest.mark.parametrize("LBox, has_tail", [(16.0, False), (4.0, True)])
def test_stacked_kernel_origin_carries_cell_average(monkeypatch, builder, LBox, has_tail):
    nGrid, nGrid_min = 16, 8
    c = nGrid // 2
    rr, vals = _steep_profile(r_max=5.0)

    new = builder(rr, vals, LBox, nGrid, nGrid_min)
    # Disable the self-term to recover the pre-fix kernel from the same code path.
    monkeypatch.setattr(_helpers, "_centre_cell_average", lambda *a, **k: 0.0)
    old = builder(rr, vals, LBox, nGrid, nGrid_min)
    monkeypatch.undo()

    if has_tail:
        # LBox < 2 * r_max: the periodic-image tail lands on the origin and must be kept.
        assert old[c, c, c] > 0.0
    else:
        # The bug: with no tail, the source's own cell was exactly empty.
        assert old[c, c, c] == 0.0

    expected = _helpers._centre_cell_average(rr, vals, LBox / nGrid)
    diff = new - old
    assert new[c, c, c] > 0.0
    assert diff[c, c, c] == pytest.approx(expected, rel=1e-12)
    # Every other cell is bit-identical to the pre-fix kernel.
    diff[c, c, c] = 0.0
    assert np.array_equal(diff, np.zeros_like(diff))
    # No hole: the origin now exceeds its face neighbours.
    assert new[c, c, c] > max(new[c + 1, c, c], new[c - 1, c, c], new[c, c + 1, c], new[c, c, c + 1])


def test_profile_to_3Dkernel_is_still_a_pure_point_sampler():
    # Guards the xHII path: the cell average must live only in the stacked builders.
    nGrid, LBox = 16, 16.0
    c = nGrid // 2
    rr, vals = _steep_profile()
    steep = profile_to_3Dkernel(interp1d(rr, vals, bounds_error=False, fill_value=0.0), nGrid, LBox)
    assert steep[c, c, c] == 0.0
    xhii = profile_to_3Dkernel(
        interp1d(rr, (rr < 0.3).astype(float), bounds_error=False, fill_value=(1.0, 0.0)), nGrid, LBox
    )
    assert xhii[c, c, c] == 1.0
    assert xhii.sum() == 1.0
