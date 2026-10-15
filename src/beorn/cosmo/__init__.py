"""Cosmology functions.

- :mod:`.background` — numpy background cosmology (Hubble, growth, distances,
  temperatures). Public names are re-exported here, so
  ``from beorn.cosmo import D, hubble`` keeps working unchanged.
- :mod:`.differentiable` — backend-generic (numpy/jax/torch), differentiable,
  GPU-capable counterparts (issue #42, Phase 1).
- :mod:`.ps_linear` — linear matter P(k): Eisenstein & Hu (default), CAMB,
  CLASS, file/tabulated. Shared by ``lpt``, ``mass_function``, ``halo_model``.
"""

from .background import (
    hubble,
    Hubble,
    comoving_distance,
    T_cmb,
    T_smooth_radio,
    read_powerspectrum,
    T_adiab,
    T_adiab_fluctu,
    E,
    D_non_normalized,
    D,
    rhoc_of_z,
    siny_ov_y,
    correlation_fct,
    Tspin_fct,
    dTb_factor,
    Tvir_to_M,
    M_to_Tvir,
    Thomson_optical_depth,
    R_of_M,
)

from .differentiable import (
    growth_factor,
    growth_rate,
    hubble_E,
)

from .ps_linear import (
    PowerSpectrum,
    get_power_spectrum,
)

__all__ = [
    # background (numpy)
    "hubble",
    "Hubble",
    "comoving_distance",
    "T_cmb",
    "T_smooth_radio",
    "read_powerspectrum",
    "T_adiab",
    "T_adiab_fluctu",
    "E",
    "D_non_normalized",
    "D",
    "rhoc_of_z",
    "siny_ov_y",
    "correlation_fct",
    "Tspin_fct",
    "dTb_factor",
    "Tvir_to_M",
    "M_to_Tvir",
    "Thomson_optical_depth",
    "R_of_M",
    # differentiable (numpy/jax/torch)
    "growth_factor",
    "growth_rate",
    "hubble_E",
    # linear power spectrum
    "PowerSpectrum",
    "get_power_spectrum",
]
