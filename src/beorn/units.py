"""Conversion helpers between BEoRN's h-unit convention and physical units.

BEoRN's internals (LPT, mass function, painting, ...) always compute in the
historical h-unit convention (Mpc/h). Quantities split into two different
handling patterns depending on whether the *user* ever directly sets them:

- ``simulation.Lbox`` is user input whose meaning depends on
  ``simulation.use_hunits`` (Mpc/h when True; physical Mpc when False, the
  default) — see :attr:`~beorn.structs.parameters.Parameters.Lbox_hunits`,
  which resolves it to the internal Mpc/h representation. This module's
  helpers are NOT used for ``Lbox`` — do not apply ``length_factor`` to it.
- Quantities that are always an *output* of internal computation (never a
  raw user input), like halo positions, are always computed in Mpc/h
  regardless of the toggle; :func:`length_factor` converts them to physical
  units for display/post-processing only (see
  :attr:`~beorn.structs.HaloCatalog.positions_physical`).

Halo-mass-valued quantities are not yet gated by ``use_hunits`` at all
(deferred, tracked separately) — :func:`mass_factor` exists for that future
work but has no call sites yet. See issue #49 for the full design history.
"""

from .constants import rhoc0 as _rhoc0_hunits


def length_factor(parameters) -> float:
    """Multiplicative factor converting a Mpc/h length to the requested unit system.

    Returns 1.0 (no-op) when ``parameters.simulation.use_hunits`` is True;
    otherwise 1/h0, converting Mpc/h to Mpc.
    """
    if parameters.simulation.use_hunits:
        return 1.0
    return 1.0 / parameters.cosmology.h0


def mass_factor(parameters) -> float:
    """Multiplicative factor converting a Msun/h mass to the requested unit system.

    Returns 1.0 (no-op) when ``parameters.simulation.use_hunits`` is True;
    otherwise 1/h0, converting Msun/h to Msun.
    """
    if parameters.simulation.use_hunits:
        return 1.0
    return 1.0 / parameters.cosmology.h0


def rhoc0_physical(parameters) -> float:
    """Critical density at z=0 in physical Msun/Mpc^3.

    ``beorn.constants.rhoc0`` is the literal 2.775e11 in h^2 Msun/Mpc^3 (i.e.
    the physical value is obtained by multiplying by h0^2). Unlike
    :func:`length_factor`/:func:`mass_factor`, this is a fully resolved
    physical constant and does not depend on ``use_hunits``.
    """
    return _rhoc0_hunits * parameters.cosmology.h0 ** 2
