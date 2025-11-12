import numpy as np
import matplotlib.pyplot as plt

from ..structs import RadiationProfiles, Parameters
from ..cosmo import T_adiab


#### Mhalo(z) from simulation from HM paper
z_array_1 = np.array([10.08083140877598, 9.705542725173212, 9.301385681293302, 8.897228637413393, 8.348729792147806, 7.684757505773671, 7.107390300230946, 6.5011547344110845, 5.92378752886836, 5.08660508083141])

Mh_z_1 = np.array([94706100.0077222, 126579158.66671978, 163154262.77379718, 222053024.50155312, 324945871.5591836, 530163041.15627587, 790019471.240894, 1243048829.5695167, 1991648063.8308542, 3689173954.4382358])

z_array_2 = np.array([12.361431870669746, 11.870669745958429, 11.379907621247115, 10.744803695150113, 9.821016166281755, 8.897228637413393, 8.146651270207853, 7.453810623556583, 6.732101616628176, 5.981524249422634, 5.057736720554274])

Mh_z_2 = np.array([98203277.90631257, 136100044.66105506, 195586368.30395073, 302214265.51783735, 570040236.7896342, 1114920997.0080914, 1852322215.6741183, 3191074972.923546, 5597981979.123267, 10000000000, 21414535638.539528])

z_array_3 = np.array([14.988452655889146, 14.642032332563511, 13.833718244803697, 12.70785219399538, 11.986143187066977, 11.23556581986143, 10.600461893764434, 9.79214780600462, 8.926096997690532, 8.146651270207853, 7.511547344110854, 6.876443418013858, 6.241339491916859, 5.6928406466512715, 5.288683602771364, 5.057736720554274])

Mh_z_3 = np.array([172274291.4769941, 218063348.75063255, 368917395.4438236, 775825016.8566794, 1243048829.5695167, 2102977594.5461233, 3493872774.7491226, 6129168695.9257145, 11987818459.583773, 21029775945.46132, 35577964903.39488, 61291686959.25715, 109488896512.76825, 178635801924.5735, 266193134612.6126, 343109759067.9875])



def plot_1D_profiles(parameters: Parameters, profiles: RadiationProfiles, mass_index: int, redshifts: list, alphas: list, label: str = None) -> None:
    """
    Plots the profiles as a function of radius. Since they are computed for a range of masses, redshifts and alpha values, the caller can specify which mass, redshift and alpha values to plot. Different redhifts are represented by different hues, while different alpha values are represented by different line styles.

    Parameters
    ----------
    parameters : Parameters
        The parameters used for the simulation.
    profiles : RadiationProfiles
        The radiation profiles computed by the solver.
    mass_index : int
        The index of the mass to plot. This is used to select the correct mass bin from the profiles.
    redshifts : list
        A list of redshifts to plot. The according index in the profiles is determined by the closest value to the specified redshift.
    alphas : list
        A list of alpha values to plot. The according index in the profiles is determined by the closest value to the specified alpha.
    """
    fig, axs = plt.subplots(1, 4, figsize=(17, 5))
    fig.suptitle(label)

    # since these are hdf5 datasets, we need to copy it to a numpy array first
    co_radial_grid = profiles.r_grid_cell[:]
    r_lyal_phys = profiles.r_lyal[:]
    zz = profiles.z_history[:]

    Mh_list = []
    actual_alpha_list = []

    for i, zi in enumerate(redshifts):
        for j, alpha_j in enumerate(alphas):
            # the user specifies the redshifts and alpha values - here we find the index lying closest to these values in the profile
            ind_z = np.argmin(np.abs(zz - zi))
            z_val = zz[ind_z]

            ind_alpha = np.argmin(np.abs(parameters.simulation.halo_mass_accretion_alpha - alpha_j))
            alpha_val = parameters.simulation.halo_mass_accretion_alpha[ind_alpha]

            # the mass history is now uniquely defined:
            Mh_i = profiles.halo_mass_bins[mass_index, ind_alpha, ind_z]
            # TODO - why the 0.68 factor? is this h?
            Mh_list.append(Mh_i / 0.68)
            actual_alpha_list.append(alpha_val)

            # some quantities are required to plot sensible profiles
            T_adiab_z = T_adiab(z_val, parameters)
            Temp_profile = profiles.rho_heat[:, mass_index, ind_alpha, ind_z] + T_adiab_z

            x_HII_profile = np.zeros((len(co_radial_grid)))
            x_HII_profile[np.where(co_radial_grid < profiles.R_bubble[mass_index, ind_alpha, ind_z])] = 1

            lyal_profile = profiles.rho_alpha[:, mass_index, ind_alpha, ind_z]  # *1.81e11/(1+zzi)

            ## plot each profile on its own axis
            # the color is determined by the redshift and alpha values
            # for increasing redshifts the color is changing from blue to red
            # for increasing alpha values the opacity is changing from faint to strong
            color = plt.cm.coolwarm((len(redshifts) - i) / len(redshifts))
            # TODO - this does not yet look good
            alpha = 1 - 0.5 * j / len(alphas)

            # the label is the same for all profiles
            label = f"$z \\sim$ {z_val:.1f}\n$M_{{h}}= {Mh_list[i]:.2e}$\n$\\alpha = {alpha_val:.2}$"


            ax = axs[0]
            ax.scatter(z_val, Mh_i / 0.68, s=150, marker='*', color=color, alpha=alpha)

            ax = axs[1]
            ax.loglog(r_lyal_phys * (1 + z_val) / 0.68, lyal_profile, lw=1.7, color=color, alpha=alpha, label=label)

            ax = axs[2]
            ax.loglog(co_radial_grid / 0.68, Temp_profile, lw=1.7, color=color, alpha=alpha)

            ax = axs[3]
            ax.semilogx(co_radial_grid / 0.68, x_HII_profile, lw=1.7, color=color, alpha=alpha)


    # plot the simulation data (and add one legend)
    ax = axs[0]
    ax.semilogy(z_array_1, Mh_z_1 / 0.68, color='gold', ls='--', lw=3, alpha=0.8)
    ax.semilogy(z_array_2, Mh_z_2 / 0.68, color='gold', ls='--', lw=3, alpha=0.8)
    ax.semilogy(z_array_3, Mh_z_3 / 0.68, color='gold', ls='--', lw=3, alpha=0.8, label='Simulation (Behroozi +20)')

    # plot our analytical data (and add one legend)
    ax.semilogy(zz, profiles.halo_mass_bins[mass_index, ind_alpha, :] / 0.68, color='gray', alpha=1, lw=2, label=f'analytical MAR\n$M_0 = {Mh_list[0]:.2e}$, $\\alpha = {actual_alpha_list[0]:.2}$')

    # style the plot
    ax.set_xlim(15, 5)
    ax.set_ylim(1.5e8, 8e12)
    ax.set_xlabel('z')
    ax.set_ylabel(r'$M_h$ [$M_{\odot}$]')
    ax.tick_params(axis="both")
    # ax.legend(loc='upper left')

    # style the other subplots and distribute the legend across all subplots by plotting an empty line
    ax = axs[1]
    ax.set_xlim(2e-1, 1e3)
    ax.set_ylim(2e-17, 1e-5)
    # ax.loglog([], [], color='C0', label=fr'$z \sim$ {z_liste[0]}, $M_{{h}}= {Mh_list[0]:.2e}$')
    ax.set_xlabel('r [cMpc]')
    ax.tick_params(axis="both")
    ax.set_ylabel(r'$\rho_{\alpha}$ [$\mathrm{pcm}^{-2}\, \mathrm{s}^{-1} \, \mathrm{Hz}^{-1}$]')
    # ax.legend()

    ax = axs[2]
    ax.set_xlim(2e-2, 1e2)
    ax.set_ylim(0.8, 5e6)
    # ax.loglog([], [], color='C1', label=fr'$z \sim$ {z_liste[1]}, $M_{{h}} = {Mh_list[1]:.2e}$')
    ax.set_xlabel('r [cMpc]')
    ax.set_ylabel(r'$\rho_{h}$ [K]')
    ax.tick_params(axis="both")
    # ax.legend()

    ax = axs[3]
    ax.set_xlim(2e-2, 1e2)
    ax.set_ylim(0, 1.2)
    # ax.semilogx([], [], color='C2', label=fr'$z \sim$ {z_liste[2]}, $M_{{h}} = {Mh_list[2]:.2e}$')
    ax.set_xlabel('r [cMpc]')
    ax.tick_params(axis="both")
    ax.set_ylabel(r'$x_{\mathrm{HII}}$')
    # ax.legend()

    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.show()


def plot_profile_alpha_dependence(ax: plt.axes, profiles: RadiationProfiles, quantity: str, mass_index: int, redshift_index: int, alphas: list, colors: list) -> None:

    # since these are hdf5 datasets, we need to copy it to a numpy array first
    co_radial_grid = profiles.r_grid_cell[:]
    r_lyal_phys = profiles.r_lyal[:]
    zz = profiles.z_history[:]

    z_val = zz[redshift_index]
    actual_alpha_list = []

    print(f"Plotting profiles at {redshift_index=}, {z_val=} => M_h = {profiles.halo_mass_bins[mass_index, 0, redshift_index]:.2e}")

    for j, alpha_j in enumerate(alphas):
        # the user specifies the redshifts and alpha values - here we find the index lying closest to these values in the profile

        ind_alpha = np.argmin(np.abs(profiles.parameters.simulation.halo_mass_accretion_alpha - alpha_j))
        alpha_val = profiles.parameters.simulation.halo_mass_accretion_alpha[ind_alpha]

        actual_alpha_list.append(alpha_val)

        # some quantities are required to plot sensible profiles
        T_adiab_z = T_adiab(z_val, profiles.parameters)
        Temp_profile = profiles.rho_heat[:, mass_index, ind_alpha, redshift_index] + T_adiab_z

        x_HII_profile = np.zeros((len(co_radial_grid)))
        x_HII_profile[np.where(co_radial_grid < profiles.R_bubble[mass_index, ind_alpha, redshift_index])] = 1

        lyal_profile = profiles.rho_alpha[:, mass_index, ind_alpha, redshift_index]  # *1.81e11/(1+zzi)

        color = colors[j]


        if quantity == 'lyal':
            ax.loglog(r_lyal_phys * (1 + z_val) / 0.68, lyal_profile, lw=1.7, color=color, alpha=0.8)
            ax.set_xlabel('r [cMpc]')
            ax.set_ylabel(r'$\rho_{\alpha}$ [$\mathrm{pcm}^{-2}\, \mathrm{s}^{-1} \, \mathrm{Hz}^{-1}$]')

        elif quantity == 'temp':
            ax.loglog(co_radial_grid / 0.68, Temp_profile, lw=1.7, color=color, alpha=0.8)

            ax.set_xlabel('r [cMpc]')
            ax.set_ylabel(r'$\rho_{h}$ [K]')


        elif quantity == 'xHII':
            ax.semilogx(co_radial_grid / 0.68, x_HII_profile, lw=1.7, color=color, alpha=0.8)
            ax.set_xlabel('r [cMpc]')
            ax.tick_params(axis="both")
            ax.set_ylabel(r'$x_{\mathrm{HII}}$')

        else:
            raise ValueError(f"Unknown quantity: {quantity}")
