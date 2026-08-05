"""Cosmology functions.

- :mod:`.background` — numpy background cosmology (Hubble, growth, distances,
  temperatures). Public names are re-exported here, so
  ``from beorn.cosmo import D, hubble`` keeps working unchanged.
- :mod:`.differentiable` — backend-generic (numpy/jax/torch), differentiable,
  GPU-capable counterparts (issue #42, Phase 1).
"""

from .background import (
    hubble,
    hubble_per_yr,
    comoving_distance,
    T_cmb,
    T_smooth_radio,
    read_powerspectrum,
    T_adiab,
    T_adiab_fluctu,
    E,
    dark_energy_density_factor,
    D_non_normalized,
    D_cpt92_non_normalized,
    D_linder2005_non_normalized,
    D_linder_cahn2007_non_normalized,
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

__all__ = [
    # background (numpy)
    "hubble",
    "hubble_per_yr",
    "comoving_distance",
    "T_cmb",
    "T_smooth_radio",
    "read_powerspectrum",
    "T_adiab",
    "T_adiab_fluctu",
    "E",
    "dark_energy_density_factor",
    "D_non_normalized",
    "D_cpt92_non_normalized",
    "D_linder2005_non_normalized",
    "D_linder_cahn2007_non_normalized",
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
]
