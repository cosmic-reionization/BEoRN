"""Matter-only halo model with NFW profiles."""
from __future__ import annotations

from .base import HaloModel
from .tracers import MatterTracer, concentration_duffy08

__all__ = ['HaloModelDM', 'concentration_duffy08']


class HaloModelDM(HaloModel):
    """Non-linear matter power spectrum: a :class:`HaloModel` with one NFW matter tracer.

    ``power_spectrum(k, z)`` returns ``P, P_1h, P_2h, P_lin`` for matter.

    Args:
        parameters:    BEoRN Parameters.
        Delta:         Halo overdensity w.r.t. the mean matter density.
        concentration: Callable ``c(M, z, h0)``; defaults to Duffy et al. 2008.
        **kwargs:      Forwarded to :class:`~.base.HaloModel`
                       (``ps_method``, ``hmf_model``, ``M_min`` ... and ``ps_var``).
    """

    def __init__(self, parameters, Delta: float = 200.0,
                 concentration=concentration_duffy08, **kwargs):
        super().__init__(parameters, include_matter=False, **kwargs)
        self.matter = self.add_tracer(MatterTracer('m', Delta=Delta, concentration=concentration))
