#!/usr/bin/env python3

"""
Computation of the meson mass from the correlator given the plateau range.
"""

from argparse import ArgumentParser
from functools import partial
import matplotlib.pyplot as plt

import pyerrors as pe
from autograd.numpy import exp

from .mesons import get_correlators_from_file, IMPLEMENTED_CHANNELS
from .meson_eff_mass import plot_eff_mass, add_mass_band


def fit_form(params, t, NT=None):
    mass = params[0]
    decay_const = params[1]
    return decay_const**2 * mass * (exp(-mass * t) + exp(-mass * (NT - t)))


def get_args():
    """
    Parse command-line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("correlator_log", help="Correlator computation log file")
    parser.add_argument(
        "--channel",
        choices=list(IMPLEMENTED_CHANNELS.keys()),
        default="ps",
        help="Which channel to fit",
    )
    parser.add_argument(
        "--plateau_start",
        required=True,
        type=int,
        help="Time slice at which plateau starts",
    )
    parser.add_argument(
        "--plateau_end",
        required=True,
        type=int,
        help="Time slice at which plateau ends",
    )
    parser.add_argument(
        "--output_filename",
        default=None,
        help="Where to output results",
    )
    parser.add_argument(
        "--plot_filename",
        default=None,
        help=(
            "Where to output an effective mass plot."
            "If not specified, no plot is generated"
        ),
    )
    parser.add_argument("--plot_styles", default=None, help="Plot style file to use")
    return parser.parse_args()


def main():
    """
    Compute the mass of a given channel
    """
    args = get_args()
    metadata, correlator = get_correlators_from_file(args.correlator_log, args.channel)
    fit_result = correlator.fit(
        partial(fit_form, NT=metadata["nt"]),
        fitrange=[args.plateau_start, args.plateau_end],
        silent=True,
    )
    for parameter in fit_result.fit_parameters:
        parameter.gamma_method()

    mass, amplitude = fit_result.fit_parameters
    if args.output_filename is None:
        print(f"mass: {mass}, " f"amplitude: {amplitude}")
    else:
        pe.input.json.dump_dict_to_json(
            {
                "mass": mass,
                "amplitude": amplitude,
            },
            args.output_filename,
            description=metadata,
        )

    if args.plot_filename is not None:
        if args.plot_styles:
            plt.style.use(args.plot_styles)
        fig = plot_eff_mass(
            correlator,
            metadata,
            plateau=(args.plateau_start, args.plateau_end),
        )
        add_mass_band(fig, mass, (args.plateau_start, args.plateau_end))
        fig.savefig(args.plot_filename)


if __name__ == "__main__":
    main()
