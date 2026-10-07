"""Mass-grid coverage of the painted halos (nonuniform-fix D9, ``solver.mass_top_check``).

The profile mass grid is defined at the reference redshift z_ref = min(solver.redshifts) and
traced back along each alpha column, M_edge(z) = M_edge(z_ref) exp(alpha (z_ref - z)). A halo of
mass M painted at redshift z with alpha column centre alpha_c therefore needs
M exp(alpha_c (z - z_ref)) below ``solver.halo_mass_bin_max``. With a tight top edge (e.g. 1e13)
high-alpha halos at high z can violate this, and the painter would stop. ``ensure_mass_top``
scans the catalogs before the profiles are computed and raises the top edge, at fixed bin width,
when needed.
"""
import logging

import numpy as np

from ..structs.helpers import bin_centers

logger = logging.getLogger(__name__)


def required_mass_top(loader, indices, alpha_edges, z_ref: float) -> dict:
    """Largest M exp(alpha_c (z - z_ref)) over the painted halos.

    Args:
        loader: provides ``redshifts`` and ``load_halo_catalog(index)`` (masses and the alphas
            the painter will use).
        indices: loader snapshot indices that will be painted.
        alpha_edges: ``solver.halo_mass_accretion_alpha``; each halo alpha is mapped to its bin
            centre, as the painter does.
        z_ref: reference redshift of the mass grid.

    Returns:
        dict with ``m_req`` (0 if no halos) and the snapshot ``z`` and ``alpha`` that set it.
    """
    edges = np.asarray(alpha_edges, dtype=float)
    centres = bin_centers(edges)
    best = {"m_req": 0.0, "z": None, "alpha": None, "mass": None}
    for i in indices:
        cat = loader.load_halo_catalog(i)
        if cat.size == 0:
            continue
        z = float(loader.redshifts[i])
        a_idx = np.clip(np.searchsorted(edges, cat.alphas, side="right") - 1, 0, centres.size - 1)
        a_c = centres[a_idx]
        m_ref = np.asarray(cat.masses, dtype=float) * np.exp(a_c * (z - z_ref))
        k = int(np.argmax(m_ref))
        if m_ref[k] > best["m_req"]:
            best = {"m_req": float(m_ref[k]), "z": z, "alpha": float(a_c[k]), "mass": float(cat.masses[k])}
    return best


def adjust_mass_top(solver_parameters, m_req: float, margin: float = 1.05) -> bool:
    """Raise ``halo_mass_bin_max`` (and ``halo_mass_nbin``) at fixed log bin width so the grid
    reaches ``margin * m_req``. The bottom edge is unchanged. Returns True if changed."""
    lo = float(solver_parameters.halo_mass_bin_min)
    hi = float(solver_parameters.halo_mass_bin_max)
    n = int(solver_parameters.halo_mass_nbin)
    width = np.log10(hi / lo) / (n - 1)
    target = margin * float(m_req)
    if target <= hi:
        return False
    k = int(np.ceil(np.log10(target / lo) / width - 1e-12))
    solver_parameters.halo_mass_nbin = k + 1
    solver_parameters.halo_mass_bin_max = float(lo * 10 ** (k * width))
    return True


def ensure_mass_top(parameters, loader, indices, margin: float = 1.05) -> dict:
    """D9: check the top edge before the profile solve and raise it if a painted halo needs it.

    No-op (returns ``{"checked": False}``) under ``solver.numerics == 'legacy'`` or when
    ``solver.mass_top_check`` is False. Must run after ``solver.redshifts`` holds the profile
    grid (z_ref = its minimum) and before the profile cache path is resolved, so the cache key
    reflects the adjusted grid.
    """
    solver = parameters.solver
    if getattr(solver, "is_legacy", True) or not getattr(solver, "mass_top_check", False):
        return {"checked": False}
    z_ref = float(np.min(solver.redshifts))
    info = required_mass_top(loader, indices, solver.halo_mass_accretion_alpha, z_ref)
    before = (float(solver.halo_mass_bin_max), int(solver.halo_mass_nbin))
    changed = adjust_mass_top(solver, info["m_req"], margin) if info["m_req"] > 0 else False
    info.update(checked=True, changed=changed, z_ref=z_ref, before=before,
                after=(float(solver.halo_mass_bin_max), int(solver.halo_mass_nbin)))
    if changed:
        logger.warning(
            "Mass-grid top edge raised from %.3e (%d edges) to %.3e (%d edges) at fixed bin width: "
            "a halo of M=%.3e at z=%.3f with alpha=%.3f needs %.3e at z_ref=%.3f.",
            before[0], before[1], solver.halo_mass_bin_max, solver.halo_mass_nbin,
            info["mass"], info["z"], info["alpha"], info["m_req"], z_ref)
    else:
        logger.info("Mass-grid top edge %.3e covers every painted halo (max need %.3e).", before[0], info["m_req"])
    return info
