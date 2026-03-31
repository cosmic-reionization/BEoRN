import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..structs import RadiationProfiles, Parameters
from ..cosmo import T_adiab


#### Mhalo(z) from simulation from HM paper
z_array_1 = np.array([10.08083140877598, 9.705542725173212, 9.301385681293302, 8.897228637413393, 8.348729792147806, 7.684757505773671, 7.107390300230946, 6.5011547344110845, 5.92378752886836, 5.08660508083141])

Mh_z_1 = np.array([94706100.0077222, 126579158.66671978, 163154262.77379718, 222053024.50155312, 324945871.5591836, 530163041.15627587, 790019471.240894, 1243048829.5695167, 1991648063.8308542, 3689173954.4382358])

z_array_2 = np.array([12.361431870669746, 11.870669745958429, 11.379907621247115, 10.744803695150113, 9.821016166281755, 8.897228637413393, 8.146651270207853, 7.453810623556583, 6.732101616628176, 5.981524249422634, 5.057736720554274])

Mh_z_2 = np.array([98203277.90631257, 136100044.66105506, 195586368.30395073, 302214265.51783735, 570040236.7896342, 1114920997.0080914, 1852322215.6741183, 3191074972.923546, 5597981979.123267, 10000000000, 21414535638.539528])

z_array_3 = np.array([14.988452655889146, 14.642032332563511, 13.833718244803697, 12.70785219399538, 11.986143187066977, 11.23556581986143, 10.600461893764434, 9.79214780600462, 8.926096997690532, 8.146651270207853, 7.511547344110854, 6.876443418013858, 6.241339491916859, 5.6928406466512715, 5.288683602771364, 5.057736720554274])

Mh_z_3 = np.array([172274291.4769941, 218063348.75063255, 368917395.4438236, 775825016.8566794, 1243048829.5695167, 2102977594.5461233, 3493872774.7491226, 6129168695.9257145, 11987818459.583773, 21029775945.46132, 35577964903.39488, 61291686959.25715, 109488896512.76825, 178635801924.5735, 266193134612.6126, 343109759067.9875])



def plot_1D_profiles(parameters: Parameters, profiles: RadiationProfiles, mass_index: int, redshifts: list, alphas: list, label: str = None, figsize: tuple = (14, 9), fontsize: int = None, labelsize: int = None, **gridspec_kw) -> None:
    """Plot 1D radiation profiles for selected masses, redshifts and alphas.

    The figure has two rows:
    - Top row: halo mass accretion history (left) and legend (right).
    - Bottom row: Lyman-alpha flux, kinetic temperature, and ionisation fraction profiles.

    Different redshifts are shown in different colours (blue → red); different
    alpha values vary in opacity.

    Args:
        parameters (Parameters): Simulation parameters used for axis labels and lookups.
        profiles (RadiationProfiles): Radiation profiles to plot.
        mass_index (int): Index selecting the mass bin to visualize.
        redshifts (list): Sequence of redshifts to plot (closest match is used).
        alphas (list): Sequence of alpha values to plot (closest match is used).
        label (str, optional): Optional suptitle for the figure.
        figsize (tuple, optional): Figure size as (width, height) in inches. Default is (14, 9).
        fontsize (int, optional): Font size for axis labels and legends.
            When ``None`` (default) the current ``rcParams`` values are used.
        labelsize (int, optional): Font size for tick labels.
            When ``None`` (default) the current ``rcParams`` values are used.
        **gridspec_kw: Extra keyword arguments forwarded to :class:`matplotlib.gridspec.GridSpec`
            (e.g. ``hspace=0.4``, ``wspace=0.3``).  When ``hspace`` or ``wspace`` is supplied,
            ``constrained_layout`` is automatically disabled so the manual spacing takes effect.
    """
    use_constrained_layout = 'hspace' not in gridspec_kw and 'wspace' not in gridspec_kw
    fig = plt.figure(figsize=figsize, constrained_layout=use_constrained_layout)
    fig.suptitle(label)

    gs = gridspec.GridSpec(2, 3, figure=fig, **gridspec_kw)
    ax_mar    = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1:])
    ax_lyal   = fig.add_subplot(gs[1, 0])
    ax_temp   = fig.add_subplot(gs[1, 1])
    ax_xhii   = fig.add_subplot(gs[1, 2])

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

            ind_alpha = np.argmin(np.abs(parameters.solver.halo_mass_accretion_alpha - alpha_j))
            alpha_val = parameters.solver.halo_mass_accretion_alpha[ind_alpha]

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
            opacity = 1 - 0.5 * j / len(alphas)

            # the label is the same for all profiles
            entry_label = f"$z \\sim$ {z_val:.1f},  $M_{{h}}= {Mh_list[i]:.2e}\\,M_\\odot$,  $\\alpha = {alpha_val:.3f}$"

            ax_mar.scatter(z_val, Mh_i / 0.68, s=150, marker='*', color=color, alpha=opacity)
            ax_lyal.loglog(r_lyal_phys * (1 + z_val) / 0.68, lyal_profile, lw=1.7, color=color, alpha=opacity, label=entry_label)
            ax_temp.loglog(co_radial_grid / 0.68, Temp_profile, lw=1.7, color=color, alpha=opacity)
            ax_xhii.semilogx(co_radial_grid / 0.68, x_HII_profile, lw=1.7, color=color, alpha=opacity)

    # plot the simulation data (and add one legend)
    ax_mar.semilogy(z_array_1, Mh_z_1 / 0.68, color='gold', ls='--', lw=3, alpha=0.8)
    ax_mar.semilogy(z_array_2, Mh_z_2 / 0.68, color='gold', ls='--', lw=3, alpha=0.8)
    ax_mar.semilogy(z_array_3, Mh_z_3 / 0.68, color='gold', ls='--', lw=3, alpha=0.8, label='Simulation (Behroozi +20)')

    # plot our analytical data (and add one legend)
    ax_mar.semilogy(zz, profiles.halo_mass_bins[mass_index, ind_alpha, :] / 0.68, color='gray', alpha=1, lw=2, label=f'Analytical MAR\n$M_0 = {Mh_list[0]:.2e}\\,M_\\odot$,  $\\alpha = {actual_alpha_list[0]:.3f}$')

    # Build kwargs dicts: only pass values when explicitly set, otherwise let rcParams govern.
    label_kw  = {'fontsize': fontsize} if fontsize is not None else {}
    tick_kw   = {'labelsize': labelsize} if labelsize is not None else {}
    legend_kw = {'fontsize': fontsize} if fontsize is not None else {}

    # style the MAR panel (no legend inside — everything goes to ax_legend)
    ax_mar.set_xlim(15, 5)
    ax_mar.set_ylim(1.5e8, 8e12)
    ax_mar.set_xlabel('z', **label_kw)
    ax_mar.set_ylabel(r'$M_h$ [$M_{\odot}$]', **label_kw)
    ax_mar.tick_params(axis='both', **tick_kw)

    # legend panel: two stacked legends — MAR model on top, redshift snapshots below
    ax_legend.axis('off')

    mar_handles, mar_labels = ax_mar.get_legend_handles_labels()
    leg_mar = ax_legend.legend(mar_handles, mar_labels, loc='upper center',
                               frameon=True, title='Halo accretion model', **legend_kw)
    ax_legend.add_artist(leg_mar)  # pin it so the second legend doesn't replace it

    snap_handles, snap_labels = ax_lyal.get_legend_handles_labels()
    ax_legend.legend(snap_handles, snap_labels, loc='lower center',
                     frameon=True, title='Redshift snapshots', **legend_kw)

    # style the profile panels
    ax_lyal.set_xlim(2e-1, 1e3)
    ax_lyal.set_ylim(2e-17, 1e-5)
    ax_lyal.set_xlabel('r [cMpc]', **label_kw)
    ax_lyal.set_ylabel(r'$\rho_{\alpha}$ [$\mathrm{pcm}^{-2}\, \mathrm{s}^{-1} \, \mathrm{Hz}^{-1}$]', **label_kw)
    ax_lyal.tick_params(axis='both', **tick_kw)

    ax_temp.set_xlim(2e-2, 1e2)
    ax_temp.set_ylim(0.8, 5e6)
    ax_temp.set_xlabel('r [cMpc]', **label_kw)
    ax_temp.set_ylabel(r'$T_k$ [K]', **label_kw)
    ax_temp.tick_params(axis='both', **tick_kw)

    ax_xhii.set_xlim(2e-2, 1e2)
    ax_xhii.set_ylim(0, 1.2)
    ax_xhii.set_xlabel('r [cMpc]', **label_kw)
    ax_xhii.set_ylabel(r'$x_{\mathrm{HII}}$', **label_kw)
    ax_xhii.tick_params(axis='both', **tick_kw)

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

        ind_alpha = np.argmin(np.abs(profiles.parameters.solver.halo_mass_accretion_alpha - alpha_j))
        alpha_val = profiles.parameters.solver.halo_mass_accretion_alpha[ind_alpha]

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
