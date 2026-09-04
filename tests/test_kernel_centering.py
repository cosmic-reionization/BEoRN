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
