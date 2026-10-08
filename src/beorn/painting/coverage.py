"""Mass-grid coverage of the painted halos (nonuniform-fix D9, ``solver.mass_top_check``).

The profile mass grid is defined at the reference redshift z_ref = min(solver.redshifts) and
traced back along each alpha column, M_edge(z) = M_edge(z_ref) exp(alpha (z_ref - z)). A halo of
mass M painted at redshift z with alpha column centre alpha_c therefore needs
M exp(alpha_c (z - z_ref)) below ``solver.halo_mass_bin_max``. With a tight top edge (e.g. 1e13)
high-alpha halos at high z can violate this, and the painter would stop. ``ensure_mass_top``
scans the catalogs before the profiles are computed and raises the top edge, at fixed bin width,
when needed.

The scan only needs per-snapshot maxima, which are cached (``cache_dir``) and reused by later runs:

- constant alpha (``source.alpha_constant``) and alpha(z) (``source.alpha_constant_z``): the
  maximum halo mass of each snapshot, read from the mass column alone
  (``loader.load_halo_masses``). The entries do not depend on alpha, so every constant-alpha
  case shares them. alpha(z) uses the largest alpha-bin centre over the painted snapshots for
  every snapshot (a bound, not the per-snapshot value).
- per-halo alphas (the merger-tree cases): the maximum mass in each alpha bin of each snapshot,
  from the full catalogs (tree fit included). The key holds only what sets the alphas
  (alpha grid, lookback, fallback, tree cache file), so tree and tree_scatter share it.
"""
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from ..structs.helpers import bin_centers

logger = logging.getLogger(__name__)

COVERAGE_CACHE_VERSION = 1


def required_mass_top(loader, indices, alpha_edges, z_ref: float) -> dict:
    """Largest M exp(alpha_c (z - z_ref)) over the painted halos, from the full catalogs.

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


# ---------------------------------------------------------------------------- alpha modes
def alpha_mode(parameters) -> str:
    """'constant', 'alpha_z' or 'per_halo' (the loader assigns alphas per halo)."""
    src = parameters.source
    if getattr(src, "alpha_constant", None) is not None:
        return "constant"
    if getattr(src, "alpha_constant_z", None) is not None:
        return "alpha_z"
    return "per_halo"


def _snapshot_alpha(parameters, z: float) -> float:
    """The alpha the loader gives every halo of a snapshot at redshift z (constant modes)."""
    src = parameters.source
    if getattr(src, "alpha_constant", None) is not None:
        return float(src.alpha_constant)
    z_table, alpha_table = src.alpha_constant_z[0], src.alpha_constant_z[1]
    return float(np.interp(z, z_table, alpha_table))


def _bin_centre(alpha_edges, alpha):
    edges = np.asarray(alpha_edges, dtype=float)
    centres = bin_centers(edges)
    return centres[np.clip(np.searchsorted(edges, alpha, side="right") - 1, 0, centres.size - 1)]


# ---------------------------------------------------------------------------- cache
def _jsonable(x):
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x if isinstance(x, (str, bool, type(None))) else repr(x)


def coverage_cache_key(parameters, loader) -> tuple[str, dict]:
    """Key of the per-snapshot maxima. Holds the catalog identity (simulation root, subbox,
    loader class) and, for per-halo alphas, everything that sets those alphas."""
    mode = "masses" if alpha_mode(parameters) != "per_halo" else "per_halo"
    subbox = getattr(loader, "subbox", None)
    d = {
        "version": COVERAGE_CACHE_VERSION,
        "kind": mode,
        "loader": type(loader).__name__,
        "file_root": str(getattr(getattr(parameters, "cosmo_sim", None), "file_root", "")),
        "subbox": None if subbox is None else repr(subbox),
    }
    if mode == "per_halo":
        tree = getattr(loader, "cached_tree", None)
        st = Path(tree).stat() if tree is not None and Path(tree).exists() else None
        d.update({
            "alpha_edges": _jsonable(np.asarray(parameters.solver.halo_mass_accretion_alpha, float)),
            "mass_accretion_lookback": _jsonable(getattr(parameters.source, "mass_accretion_lookback", None)),
            "alpha_fallback": _jsonable(getattr(parameters.source, "alpha_fallback", None)),
            "tree_cache": None if tree is None else [str(tree), st.st_size if st else None,
                                                     st.st_mtime_ns if st else None],
        })
    key = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
    return key, d


def _load_cache(path: Path, desc: dict) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        logger.warning("Ignoring unreadable coverage cache %s (%s).", path, err)
        return {}
    if data.get("key") != desc:
        logger.warning("Coverage cache %s has a different key; ignoring it.", path)
        return {}
    return data.get("snapshots", {})


def _save_cache(path: Path, desc: dict, snapshots: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"key": desc, "snapshots": snapshots}, indent=1, sort_keys=True))
    tmp.replace(path)


def snapshot_maxima(parameters, loader, indices, cache_dir=None) -> dict:
    """Per-snapshot maxima for the painted ``indices``, from the cache where present.

    Returns {str(index): entry}; entry has ``z``, ``n`` (halos) and either ``m_max`` (constant
    modes) or ``m_max_by_alpha_bin`` (per-halo mode; 0 for empty bins)."""
    key, desc = coverage_cache_key(parameters, loader)
    path = None if cache_dir is None else Path(cache_dir) / f"mass_coverage_{key}.json"
    cached = _load_cache(path, desc)
    out, n_new = {}, 0
    per_halo = desc["kind"] == "per_halo"
    edges = np.asarray(parameters.solver.halo_mass_accretion_alpha, float)
    for i in indices:
        z = float(loader.redshifts[i])
        e = cached.get(str(int(i)))
        if e is not None and abs(e["z"] - z) < 1e-6:
            out[str(int(i))] = e
            continue
        if per_halo:
            cat = loader.load_halo_catalog(i)
            m = np.asarray(cat.masses, float)
            b = np.clip(np.searchsorted(edges, cat.alphas, side="right") - 1, 0, edges.size - 2)
            mb = np.zeros(edges.size - 1)
            if m.size:
                np.maximum.at(mb, b, m)
            e = {"z": z, "n": int(m.size), "m_max_by_alpha_bin": mb.tolist()}
        else:
            m = np.asarray(loader.load_halo_masses(i), float)
            e = {"z": z, "n": int(m.size), "m_max": float(m.max()) if m.size else 0.0}
        out[str(int(i))] = e
        cached[str(int(i))] = e
        n_new += 1
    if n_new:
        _save_cache(path, desc, cached)
    logger.info("Mass-coverage maxima: %d snapshot(s) from cache, %d scanned (%s)%s.",
                len(out) - n_new, n_new, desc["kind"], "" if path is None else f", cache {path}")
    return out


def required_mass_top_from_maxima(parameters, maxima: dict, z_ref: float) -> dict:
    """The required top edge from :func:`snapshot_maxima`.

    constant: alpha_c of the constant alpha. alpha_z: the largest alpha-bin centre over the
    painted snapshots, applied to every snapshot (bound). per_halo: exact per alpha bin.
    Also returns, for alpha_z, the exact per-snapshot value and the crude global bound
    (largest mass of any snapshot x exp(alpha_max (largest z - z_ref)))."""
    edges = np.asarray(parameters.solver.halo_mass_accretion_alpha, float)
    mode = alpha_mode(parameters)
    entries = [e for e in maxima.values() if e["n"] > 0]
    info = {"m_req": 0.0, "z": None, "alpha": None, "mass": None, "mode": mode}
    if not entries:
        return info
    z = np.array([e["z"] for e in entries])
    if mode == "per_halo":
        centres = bin_centers(edges)
        mb = np.array([e["m_max_by_alpha_bin"] for e in entries])          # (snap, bin)
        req = mb * np.exp(centres[None, :] * (z[:, None] - z_ref))
        s, b = np.unravel_index(int(np.argmax(req)), req.shape)
        info.update(m_req=float(req[s, b]), z=float(z[s]), alpha=float(centres[b]), mass=float(mb[s, b]))
        return info
    m = np.array([e["m_max"] for e in entries])
    a_c = np.array([_bin_centre(edges, _snapshot_alpha(parameters, zz)) for zz in z])
    exact = m * np.exp(a_c * (z - z_ref))
    if mode == "constant":
        k = int(np.argmax(exact))
        info.update(m_req=float(exact[k]), z=float(z[k]), alpha=float(a_c[k]), mass=float(m[k]))
        return info
    a_max = float(a_c.max())
    bound = m * np.exp(a_max * (z - z_ref))
    k = int(np.argmax(bound))
    ke = int(np.argmax(exact))
    info.update(m_req=float(bound[k]), z=float(z[k]), alpha=a_max, mass=float(m[k]),
                exact_m_req=float(exact[ke]), exact_z=float(z[ke]), exact_alpha=float(a_c[ke]),
                global_bound=float(m.max() * np.exp(a_max * (z.max() - z_ref))),
                alpha_range=(float(a_c.min()), a_max))
    return info


def ensure_mass_top(parameters, loader, indices, margin: float = 1.05, cache_dir=None) -> dict:
    """D9: check the top edge before the profile solve and raise it if a painted halo needs it.

    No-op (returns ``{"checked": False}``) under ``solver.numerics == 'legacy'`` or when
    ``solver.mass_top_check`` is False. Must run after ``solver.redshifts`` holds the profile
    grid (z_ref = its minimum) and before the profile cache path is resolved, so the cache key
    reflects the adjusted grid. ``cache_dir`` holds the per-snapshot maxima (see the module
    docstring); None scans without caching.
    """
    solver = parameters.solver
    if getattr(solver, "is_legacy", True) or not getattr(solver, "mass_top_check", False):
        return {"checked": False}
    z_ref = float(np.min(solver.redshifts))
    maxima = snapshot_maxima(parameters, loader, indices, cache_dir=cache_dir)
    info = required_mass_top_from_maxima(parameters, maxima, z_ref)
    before = (float(solver.halo_mass_bin_max), int(solver.halo_mass_nbin))
    changed = adjust_mass_top(solver, info["m_req"], margin) if info["m_req"] > 0 else False
    info.update(checked=True, changed=changed, z_ref=z_ref, before=before,
                after=(float(solver.halo_mass_bin_max), int(solver.halo_mass_nbin)))
    if info["mode"] == "alpha_z":
        logger.info(
            "alpha(z) top-edge bound: alpha-bin centres %.3f-%.3f over the painted snapshots; with "
            "alpha_max=%.3f per snapshot the need is %.3e (z=%.3f); exact per-snapshot alpha gives %.3e "
            "(z=%.3f); largest mass x exp(alpha_max (z_max - z_ref)) would be %.3e.",
            info["alpha_range"][0], info["alpha_range"][1], info["alpha"], info["m_req"], info["z"],
            info["exact_m_req"], info["exact_z"], info["global_bound"])
    if changed:
        logger.warning(
            "Mass-grid top edge raised from %.3e (%d edges) to %.3e (%d edges) at fixed bin width: "
            "a halo of M=%.3e at z=%.3f with alpha=%.3f needs %.3e at z_ref=%.3f.",
            before[0], before[1], solver.halo_mass_bin_max, solver.halo_mass_nbin,
            info["mass"], info["z"], info["alpha"], info["m_req"], z_ref)
    else:
        logger.info("Mass-grid top edge %.3e covers every painted halo (max need %.3e).", before[0], info["m_req"])
    return info
