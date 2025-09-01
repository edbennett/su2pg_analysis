#!/usr/bin/env python3

"""
Plotting the effective mass for a single channel's meson correlator.
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt

from .mesons import get_correlators_from_file, LATEX_DESCRIPTIONS
from .plots import save_or_show


def get_args():
    """
    Parse command-line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("correlator_log", help="Correlator computation log file")
    parser.add_argument("--channel", choices=list(LATEX_DESCRIPTIONS.keys()), default="ps", help="Which channel to fit")
    parser.add_argument(
        "--output_filename",
        default=None,
        help="Where to output the resulting plot",
    )
    parser.add_argument("--plot_styles", default=None, help="Plot style file to use")
    return parser.parse_args()


def get_slice(plateau, metadata):
    """
    Take a given start and end value for a plateau,
    and return an appropriate slice of indices to plot to show the plateau in context
    """
    plateau_start, plateau_end = plateau
    max_time = metadata["nt"] // 2
    if plateau_end is None:
        plateau_end = max_time
    if plateau_start is None:
        return slice(0, plateau_end)

    desired_context = (plateau_end - plateau_start) // 2
    return slice(
        max(min(plateau_start, 2), plateau_start - desired_context),
        min(max_time, plateau_end + desired_context)
    )


def plot_eff_mass(correlator, metadata, plateau=(None, None)):
    """
    Plot the effective mass of a correlator.
    Optionally,
    restrict to a given plateau
    """
    fig, ax = plt.subplots(layout="constrained")

    ax.set_xlabel("$t/a$")
    ax.set_ylabel(r"$am_{\mathrm{eff}}$")

    eff_mass = correlator.m_eff(variant="cosh")
    eff_mass.gamma_method()
    timeslice, eff_mass_value, eff_mass_uncertainty = eff_mass.plottable()

    plateau_slice = get_slice(plateau, metadata)
    ax.errorbar(timeslice[plateau_slice], eff_mass_value[plateau_slice], yerr=eff_mass_uncertainty[plateau_slice], ls="none")

    ax.set_title(
        (
            "${nt}\\times{nx}\\times{ny}\\times{nz}$, "
            "$\\beta={beta}$, $am_0={mass}$, ${channel_slug}$"
        ).format(
            **metadata,
            channel_slug=LATEX_DESCRIPTIONS[metadata["channel"]],
        )
    )

    return fig


def add_mass_band(fig, mass, plateau):
    """
    Plot a horizontal band for a given mass on the first Axes in the given figure,
    and vertical lines indicating the plateau region in which this was fitted.
    """
    ax = fig.get_axes()[0]
    plateau_start, plateau_end = plateau
    ax.axvline(plateau_start - 0.5, dashes=(2, 3))
    ax.axvline(plateau_end - 0.5, dashes=(2, 3))
    ax.axhspan(mass.value - mass.dvalue, mass.value + mass.dvalue, alpha=0.3, color="black")
    ax.axhline(mass.value, color="black")


def main():
    """
    Compute the mass of a given channel
    """
    args = get_args()
    if args.plot_styles is not None:
        plt.style.use(args.plot_styles)

    metadata, correlator = get_correlators_from_file(args.correlator_log, args.channel)
    save_or_show(plot_eff_mass(correlator, metadata), args.output_filename)


if __name__ == "__main__":
    main()
