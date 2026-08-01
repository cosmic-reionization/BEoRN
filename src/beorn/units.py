"""Conversion helpers between BEoRN's h-unit convention and physical units.

BEoRN's internals (LPT, mass function, painting, ...) all compute in the
historical h-unit convention (lengths in Mpc/h, masses in Msun/h in some
places), gated by :attr:`~beorn.structs.parameters.SimulationParameters.use_hunits`
(default ``True``, preserving existing behaviour). These helpers are meant to
be applied at the boundary — where a user-facing quantity is constructed from
or converted to a physical (non-h) value — not scattered through validated
internal numerics. See issue #49 for the full design.
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
