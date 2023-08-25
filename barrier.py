"""
This module contains functions for plotting barrier figures
"""

import numpy as np
import matplotlib.pyplot as plt
from . import utils as ut


def _evolve_trans_mat(J, Jz_expect, trans, allowed_trans="forwards",
                      scale_trans=True, normalise=True):
    """
    Scales transition matrix by "amount" coming into each state
    and removes backwards or downwards transitions

    Parameters
    ----------
    J : float
        J quantum number
    Jz_expect : np.ndarray
        1D array of <Jz> in eigenbasis of HCF
    trans : np.ndarray
        Matrix representation of magnetic transition dipole moment
        operator
    allowed_trans : str {'forwards','all'}
        Which transitions should be plotted:
            forwards: Only those which move up and over barrier
            all : All transitions
    scale_trans : bool, default True
        If true, scale all outgoing transitions from a state by
        amount coming in.
    normalise : bool, default True
        If true, normalise all transitions from a state by their sum

    Returns
    -------
    np.ndarray
        Matrix representation of magnetic transition dipole moment
        operator after scaling
    """

    # Calculate number of states
    n_states = int(2 * J + 1)

    # Remove self transitions
    np.fill_diagonal(trans, 0.)

    # Remove all transitions backwards over the barrier
    # or downwards between states
    if allowed_trans == "forwards":
        for i in np.arange(n_states):  # from
            for f in np.arange(n_states):  # to
                if Jz_expect[i] > Jz_expect[f]:
                    trans[f, i] = 0.  # No backwards or downwards steps

    # Normalise each column so transition probability is a fraction of 1
    if normalise:
        for i in np.arange(n_states):
            total = 0.
            total = sum(trans[:, i])
            if total > 0.:
                trans[:, i] = trans[:, i] / total

    # Find indexing which relates the current arrrangement of the array
    # Jz_expect to the arrangement it would
    # have if it was written in descending order (largest first)
    # This is done because our pathway consists only of transitions which
    # increase (-ve to eventually +ve) <Jz>
    index = Jz_expect.argsort()

    # Scale transition probability by amount coming into each state
    # Assume unit "population" of ground state (Jz=-J)
    # i.e. trans[:,index[0]] is already 1
    if scale_trans:
        for ind in index:
            if ind == 0:
                continue
            else:
                # scale outward by inward
                trans[:, ind] *= np.sum(trans[ind, :])

    # Scale matrix to be a percentage
    trans = 100. * trans

    # Find transitions with >1% probability
    # write their indices to an array along with the probabilty as a decimal
    # this is used to set the transparency of the arrows on the plot
    num_trans = 0
    output_trans = []
    for row in np.arange(n_states):
        for col in np.arange(n_states):
            if trans[row, col] > 1.:
                alpha = float(trans[row, col] / 100.0)
                if alpha > 1.:
                    alpha = 1.
                output_trans.append([row, col, alpha])
                num_trans += 1

    return output_trans, num_trans


def barrier_figure(J, energies, Jz_expect, trans=False, ax_in=False,
                   trans_colour="#ff0000", state_colour="black",
                   yax2=False, yax2_conv=1.4, figsize=[7, 5.5],
                   show=False, save=True, save_name="barrier.svg",
                   scale_trans=True, allowed_trans="forwards",
                   normalise_trans=True, levels_name="",
                   xlabel=r"$\langle \ \hat{J}_{z} \ \rangle$",
                   ylabel=r"Energy (cm$^{-1}$)", yax2_label="Energy (K)"):
    """
    Plots barrier figure with transition intensities from user provided matrix
    Y axis is Energy in cm-1, x axis is <Jz> of each state
    Arrows are transitions with intensity specified by specified by trans array

    Parameters
    ----------
    J : float
        J or L or S quantum number
    energies : array_like
        List of state energies
    Jz_expect : array_like
        List of <Jz> for each state
    trans : np.ndarray
        Matrix of transition probabilities between states
    ax_in : pyplot axis object
        Axis to use for plot
    trans_colour : str, default "#ff0000" (red)
        Hex code or name specifying arrow colours
    state_colour : str, default "black"
        Hex code or name specifying state colours
    yax2 : bool, default True
        If True use secondary y (energy) axis
    yax2_conv : float, default 1.4 (cm-1 --> K)
        conversion factor from primary to secondary y axis
    figsize : array_like, default [7, 5.5]
        Size of figure [width, height] in inches
    show : bool, default False
        If True, show plot on screen - disabled with ax_in
    save : bool, default True
        If True, save plot to file - disabled with ax_in
    save_name : str, default "barrier.svg"
        Filename for saved image
    allowed_trans : str {'forwards','all'}
        Which transitions should be plotted:
            forwards: Only those which move up and over barrier
            all : All transitions
    normalise_trans : bool, default True
        If True, normalise all transitions out of a state by their sum
    scale_trans : bool, default True
        If true, scale all outgoing transitions from a state by amount
        coming in.
    levels_name : str, default ""
        Legend label name for energy levels
    xlabel : str, default "hat{J}_z"
        Plot x label
    ylabel : str, default "Energy (cm-1)"
        Plot y label
    yax2_label : str, default "Energy (K)"
        Label for secondary y (energy) axis

    Returns
    -------
    pyplot figure object
        Figure window handle
    pyplot axis object
        Axes for current plot
    """

    # Create plot and axes
    if not ax_in:
        fig, ax = plt.subplots(1, 1, sharey='all', figsize=figsize)
    else:
        fig = None
        ax = ax_in

    if yax2:
        ax2 = ax.twinx()
        axes = [ax, ax2]
    else:
        axes = [ax]

    # Draw energy level lines
    ax.plot(
        Jz_expect,
        energies,
        marker='_',
        markersize='25',
        mew='2.5',
        linewidth=0,
        color=state_colour,
        label=levels_name
    )

    # Plot transition arrows
    if isinstance(trans, np.ndarray):

        # Evolve transition matrix and find allowed transitions
        output_trans, num_trans = _evolve_trans_mat(
            J,
            Jz_expect,
            trans,
            allowed_trans=allowed_trans,
            normalise=normalise_trans,
            scale_trans=scale_trans
        )

        np.savetxt("inputtrans.dat", trans)
        np.savetxt("outputtrans.dat", output_trans)

        # Final <Jz>
        Jz_expect_final = [
            Jz_expect[output_trans[row][1]]
            for row in range(num_trans)
        ]

        # Difference between initial and final <Jz>
        Jz_expect_diff = [
            Jz_expect[output_trans[row][0]]-Jz_expect[output_trans[row][1]]
            for row in range(num_trans)
        ]

        # Final energies
        energies_final = [
            energies[output_trans[row][1]]
            for row in range(num_trans)
        ]

        # Difference between initial and final energies
        energies_diff = [
            energies[output_trans[row][0]] - energies[output_trans[row][1]]
            for row in range(num_trans)
        ]

        # Alpha channel values
        alphas = [output_trans[row][2] for row in range(num_trans)]

        # Make colours array
        # Columns are red, green, blue, alpha
        t_rgba_colors = np.zeros((num_trans, 4))

        # Convert user hex to rgb
        t_rgba_colors[:, 0:3] = ut.hex_to_rgb(trans_colour)

        t_rgba_colors[:, 3] = np.asarray(alphas)

        # Draw lines between levels
        ax.quiver(
            Jz_expect_final,
            energies_final,
            Jz_expect_diff,
            energies_diff,
            scale_units='xy',
            angles='xy',
            scale=1,
            color=t_rgba_colors
        )

    # Set x axis options
    ax.set_xlabel(xlabel)
    ax.tick_params(axis='both', which='both', length=2.0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))

    # Set y axis options for cm-1
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(plt.MaxNLocator(7))

    # Set y axis options for K
    if yax2:
        ax2.set_ylabel(yax2_label)
        ax2.set_ylim(
            ax.get_ylim()[0] * yax2_conv,
            ax.get_ylim()[1] * yax2_conv
        )
        ax2.yaxis.set_major_locator(plt.MaxNLocator(7))

    # Set axis limits
    ax.set_xlim([-J * 1.1, J * 1.1])

    # Set number and position of x axis ticks
    ax.set_xticks(np.arange(-J, J + 1, 1))

    # Set x axis tick labels
    labels = []

    # Fractions if J non-integer
    if J % 2 != 0:
        for it in np.arange(0, int(2 * J + 1)):
            labels.append(str(-int(2 * J) + 2 * it) + '/2')
    else:
        for it in np.arange(0, int(2 * J + 1)):
            labels.append(str(-int(J) + it))

    ax.set_xticklabels(labels, rotation=45)

    if not ax_in:
        fig.tight_layout()
        # Save or show plot
        if save:
            fig.savefig(save_name, dpi=500)
        if show:
            plt.show()

    return fig, axes
