"""Halo-model statistics for any halo-sourced field (issue #38).

The grid pipeline paints per-halo profiles around simulated haloes; this
subpackage uses profiles analytically in the halo model, giving P(k) in
milliseconds per redshift without a simulation box.

Classes
-------
HaloModel       — generic: linear P(k), n(M), b(M), any set of tracers,
                  all auto/cross 1h/2h spectra (bispectrum: planned)
HaloModelDM     — HaloModel with an NFW matter tracer only
HaloModel21cm   — HaloModel with matter + ion/alpha/heat tracers from BEoRN's
                  radiation profiles, combined into P_21
Tracers         — Tracer (base), MatterTracer, ProfileTracer, TopHatTracer, FunctionTracer

Quick start::

    import numpy as np
    from beorn.structs import Parameters
    from beorn.halo_model import HaloModel, HaloModel21cm, FunctionTracer

    param = Parameters()
    out = HaloModel21cm(param).compute(k=np.logspace(-2, 1, 40), z=[8, 10, 12])

    hm = HaloModel(param, ps_method='camb')                 # generic, any tracer
    hm.add_tracer(FunctionTracer('g', my_window))
    hm.power_spectra(k, z=1.0)['P_m_g']
"""
from .base import HaloModel
from .bias import halo_bias
from .matter import HaloModelDM
from .profiles import nfw_ft, profile_ft, profile_ft_kernel, tophat_ft, tophat_window
from .signal21cm import FIELDS, HaloModel21cm
from .tracers import (
    BinnedTracer,
    FunctionTracer,
    MatterTracer,
    ProfileTracer,
    TopHatTracer,
    Tracer,
    bins_from_edges,
    concentration_duffy08,
    log_mass_bins,
)

__all__ = [
    'FIELDS',
    'BinnedTracer',
    'FunctionTracer',
    'HaloModel',
    'HaloModel21cm',
    'HaloModelDM',
    'MatterTracer',
    'ProfileTracer',
    'TopHatTracer',
    'Tracer',
    'bins_from_edges',
    'concentration_duffy08',
    'halo_bias',
    'log_mass_bins',
    'nfw_ft',
    'profile_ft',
    'profile_ft_kernel',
    'tophat_ft',
    'tophat_window',
]
