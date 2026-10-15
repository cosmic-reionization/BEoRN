"""Backwards-compatible alias for :mod:`beorn.cosmo.ps_linear`.

The linear power spectrum models moved to ``beorn.cosmo.ps_linear`` so
that ``lpt``, ``mass_function`` and ``halo_model`` share one implementation.
"""
from ..cosmo.ps_linear import *  # noqa: F401,F403
from ..cosmo.ps_linear import __all__  # noqa: F401
