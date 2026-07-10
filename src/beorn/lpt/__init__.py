"""LPT (Lagrangian Perturbation Theory) subpackage.

Provides initial-conditions generation and displacement-field solvers
at 1LPT (Zel'dovich), 2LPT, and (stub) 3LPT order, together with
pluggable power spectrum models and compute backends.

Quick start::

    from beorn import lpt
    from beorn.structs import Parameters

    param = Parameters()
    solver = lpt.SecondOrderLPT(param, seed=42)
    delta = solver.get_density(z=10.0)

Power spectrum methods: 'eisenstein_hu' (default), 'eisenstein_hu_wiggle', 'boltzmann', 'tabulated'.
Backends: 'numpy' (default), 'torch', 'jax'.
"""
from .linear_power import (
    PowerSpectrum,
    EisensteinHu,
    BoltzmannSolver,
    TabulatedPowerSpectrum,
    DiscoEB,
    get_power_spectrum,
    transfer_eh_nowiggle,
    sigma8_normalisation,
    pk_eh_nowiggle,
)
from .lpt import (
    LPTBase,
    ZeldovichApproximation,
    SecondOrderLPT,
    ThirdOrderLPT,
)
from .backends import LPTBackend, NumpyBackend, TorchBackend, JaxBackend, get_backend
from .chmf import CHMF, CHMFSampler, conditional_dndlnm_diff

__all__ = [
    # power spectrum
    'PowerSpectrum',
    'EisensteinHu',
    'BoltzmannSolver',
    'TabulatedPowerSpectrum',
    'DiscoEB',
    'get_power_spectrum',
    # differentiable pure functions (numpy/jax/torch)
    'transfer_eh_nowiggle',
    'sigma8_normalisation',
    'pk_eh_nowiggle',
    # LPT solvers
    'LPTBase',
    'ZeldovichApproximation',
    'SecondOrderLPT',
    'ThirdOrderLPT',
    # backends
    'LPTBackend',
    'NumpyBackend',
    'TorchBackend',
    'JaxBackend',
    'get_backend',
    # CHMF halo sampling
    'CHMF',
    'CHMFSampler',
    'conditional_dndlnm_diff',
]
