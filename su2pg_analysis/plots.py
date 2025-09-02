#!/usr/bin/env python3

"""
Shared tools for plotting
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
from .io import get_data


def get_args(description, extra_options):
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "data_filenames",
        metavar="datafile",
        help="Data files to read and plot",
        nargs="+",
    )
    parser.add_argument("--output_filename", default=None, help="Where to put the plot")
    parser.add_argument("--plot_styles", default=None, help="Plot style file to use")
    for name, kwargs in extra_options.items():
        parser.add_argument(name, **kwargs)
    return parser.parse_args()


def save_or_show(fig, output_filename):
    """
    Either save a figure to disk,
    display it on screen,
    or flush it from the buffer.
    """
    if output_filename is None:
        plt.show()
    elif output_filename != "/dev/null":
        fig.savefig(output_filename)
    plt.close(fig)


def basic_plot(plot_callback, description=None, extra_options={}):
    """
    Perform a simple plot,
    where one set of data is read,
    and one plot is generated.
    """
    args = get_args(description, extra_options)
    if args.plot_styles is not None:
        plt.style.use(args.plot_styles)
    data = get_data(args.data_filenames)
    fig = plot_callback(
        data,
        *[getattr(args, option.lstrip("-")) for option in extra_options],
    )
    save_or_show(fig, args.output_filename)


def split_errors(data):
    """
    Splits a collection of numbers into a list of central values,
    and a list of uncertainties if the numbers carry them
    (or None otherwise).
    """
    if any(hasattr(datum, "gamma_method") for datum in data):
        return tuple(
            zip(
                *[
                    (datum.value, datum.dvalue)
                    if hasattr(datum, "dvalue")
                    else (datum, None)
                    for datum in data
                ]
            )
        )
    else:
        return data, None


def errorbar_pyerrors(ax, x_data, y_data, **kwargs):
    """
    Plot y_data against x_data on ax,
    where x_data and y_data may contain numbers with or without errors.
    """
    x_values, x_errors = split_errors(x_data)
    y_values, y_errors = split_errors(y_data)
    return ax.errorbar(
        x_values,
        y_values,
        xerr=x_errors,
        yerr=y_errors,
        ls="none",
        **kwargs,
    )
