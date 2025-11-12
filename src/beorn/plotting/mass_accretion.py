import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)

from ..structs import RadiationProfiles
from ..radiation_profiles.massaccretion import mass_accretion


# TODO - make better
def plot_halo_mass_evolution(
        profiles: RadiationProfiles,
        mass_bin: int = None,
    ):


    if mass_bin is None:
        # use the middle mass bin
        mass_bin = profiles.halo_mass_bins.size // 2


    m0_string = fr"$M_0 = {profiles.parameters.simulation.halo_mass_bin_centers[mass_bin]:.2g} M_{{\odot}}$"

    masses, mass_derivatives = mass_accretion(
        profiles.parameters,
        profiles.z_history,
        profiles.parameters.simulation.halo_mass_bin_centers,
        profiles.parameters.simulation.halo_mass_accreation_alpha_bin_centers
    )
    alphas = profiles.parameters.simulation.halo_mass_accretion_alpha

    plt.figure()
    plt.xscale('log')
    plt.xlabel('Redshift z')
    plt.ylabel('Halo Mass [Msol]')
    plt.yscale('log')
    plt.title(f"Halo Mass - {m0_string}")
    for i in range(len(alphas) - 1):
        # plot a given starting mass for all alpha values
        plt.plot(profiles.z_history, masses[mass_bin, i, :], label=f"$\\alpha = {alphas[i]:.2f}$")
    plt.legend()
    plt.show()


    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Halo Mass derivative - {m0_string}")
    for i in range(len(alphas) - 1):
        # plot a given starting mass for all alpha values
        plt.plot(profiles.z_history, mass_derivatives[mass_bin, i, :], label=f"$\\alpha = {alphas[i]:.2f}$")
    plt.show()
