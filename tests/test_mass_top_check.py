"""nonuniform-fix D9: the top edge of the mass grid is raised before the profile solve if needed."""
import sys
import types

if "MAS_library" not in sys.modules:
    sys.modules["MAS_library"] = types.SimpleNamespace(MASL=None)

from types import SimpleNamespace

import numpy as np
import pytest

from beorn.painting.coverage import adjust_mass_top, ensure_mass_top, required_mass_top
from beorn.structs.parameters import Parameters


class FakeLoader:
    def __init__(self, redshifts, catalogs):
        self.redshifts = np.asarray(redshifts)
        self._cats = catalogs

    def load_halo_catalog(self, i):
        m, a = self._cats[i]
        return SimpleNamespace(masses=np.asarray(m, float), alphas=np.asarray(a, float), size=len(m))


def params(top=1e13, nbin=72):
    p = Parameters()
    p.solver.halo_mass_bin_min = 1e9
    p.solver.halo_mass_bin_max = top
    p.solver.halo_mass_nbin = nbin
    p.solver.halo_mass_accretion_alpha = np.linspace(0.0, 1.5, 16)
    p.solver.redshifts = np.array([20.0, 12.0, 8.0, 5.5])
    return p


def test_required_top_uses_alpha_bin_centres():
    loader = FakeLoader([12.0], [([1e9, 2e9], [0.62, 0.30])])
    info = required_mass_top(loader, [0], np.linspace(0.0, 1.5, 16), z_ref=5.5)
    # alpha 0.62 -> bin [0.6, 0.7) centre 0.65
    assert info["m_req"] == pytest.approx(1e9 * np.exp(0.65 * 6.5))
    assert info["alpha"] == pytest.approx(0.65)


def test_adjust_keeps_bin_width_and_bottom_edge():
    p = params()
    width = np.log10(1e13 / 1e9) / 71
    assert adjust_mass_top(p.solver, 3e13) is True
    new_width = np.log10(p.solver.halo_mass_bin_max / p.solver.halo_mass_bin_min) / (p.solver.halo_mass_nbin - 1)
    assert new_width == pytest.approx(width, rel=1e-12)
    assert p.solver.halo_mass_bin_min == 1e9
    assert p.solver.halo_mass_bin_max >= 1.05 * 3e13
    assert adjust_mass_top(p.solver, 1e12) is False


def test_ensure_raises_only_when_needed_and_skips_legacy():
    loader = FakeLoader([20.0, 8.0], [([1e9], [1.2]), ([3e11], [0.5])])
    p = params()
    info = ensure_mass_top(p, loader, [0, 1])
    assert info["changed"] and p.solver.halo_mass_bin_max > 1e13     # 1e9 * exp(1.25*14.5) >> 1e13
    covered = FakeLoader([8.0], [([3e11], [0.5])])
    q = params()
    assert ensure_mass_top(q, covered, [0])["changed"] is False
    leg = params()
    leg.solver.numerics = "legacy"
    assert ensure_mass_top(leg, loader, [0, 1]) == {"checked": False}
    assert leg.solver.halo_mass_bin_max == 1e13


# ---------------------------------------------------------------------------- cache + mass-only
class CountingLoader(FakeLoader):
    def __init__(self, redshifts, catalogs):
        super().__init__(redshifts, catalogs)
        self.catalog_reads, self.mass_reads = 0, 0

    def load_halo_catalog(self, i):
        self.catalog_reads += 1
        return super().load_halo_catalog(i)

    def load_halo_masses(self, i):
        self.mass_reads += 1
        return np.asarray(self._cats[i][0], float)


def test_constant_alpha_reads_masses_only_and_caches(tmp_path):
    p = params()
    p.source.alpha_constant = 0.4979
    p.solver.halo_mass_accretion_alpha = np.array([0.4479, 0.5479])
    loader = CountingLoader([12.0, 8.0], [([1e9, 2e10], [0, 0]), ([3e11], [0])])
    info = ensure_mass_top(p, loader, [0, 1], cache_dir=tmp_path)
    assert loader.catalog_reads == 0 and loader.mass_reads == 2
    assert info["m_req"] == pytest.approx(max(2e10 * np.exp(0.4979 * 6.5), 3e11 * np.exp(0.4979 * 2.5)))
    q = params()
    q.source.alpha_constant = 0.4979
    q.solver.halo_mass_accretion_alpha = np.array([0.4479, 0.5479])
    again = CountingLoader([12.0, 8.0], [([1e9, 2e10], [0, 0]), ([3e11], [0])])
    assert ensure_mass_top(q, again, [0, 1], cache_dir=tmp_path)["m_req"] == pytest.approx(info["m_req"])
    assert again.mass_reads == 0 and again.catalog_reads == 0


def test_alpha_z_uses_the_largest_alpha_for_every_snapshot(tmp_path):
    p = params()
    p.source.alpha_constant_z = (np.array([5.0, 20.0]), np.array([0.32, 0.52]))
    p.solver.halo_mass_accretion_alpha = np.linspace(0.3, 0.6, 7)        # centres 0.325 ... 0.575
    loader = CountingLoader([12.0, 6.0], [([1e10], [0]), ([1e12], [0])])
    info = ensure_mass_top(p, loader, [0, 1], cache_dir=tmp_path)
    a12 = 0.32 + 0.2 * (12 - 5) / 15                                      # 0.4133 -> bin [0.40, 0.45)
    assert info["alpha"] == pytest.approx(0.425)
    assert info["m_req"] == pytest.approx(max(1e10 * np.exp(0.425 * 6.5), 1e12 * np.exp(0.425 * 0.5)))
    assert info["exact_m_req"] <= info["m_req"]
    assert info["global_bound"] == pytest.approx(1e12 * np.exp(0.425 * 6.5))
    assert 0.40 <= a12 < 0.45


def test_tree_and_tree_scatter_share_the_cache(tmp_path):
    from beorn.painting.coverage import coverage_cache_key
    a, b = params(), params()
    b.source.f_st = 0.02
    b.source.f_st_paint_distribution = "lognormal"
    loader = CountingLoader([12.0], [([1e9], [0.5])])
    assert coverage_cache_key(a, loader) == coverage_cache_key(b, loader)
    ensure_mass_top(a, loader, [0], cache_dir=tmp_path)
    second = CountingLoader([12.0], [([1e9], [0.5])])
    ensure_mass_top(b, second, [0], cache_dir=tmp_path)
    assert loader.catalog_reads == 1 and second.catalog_reads == 0
    c = params()
    c.solver.halo_mass_accretion_alpha = np.linspace(0.0, 1.6, 17)
    assert coverage_cache_key(c, loader)[0] != coverage_cache_key(a, loader)[0]
